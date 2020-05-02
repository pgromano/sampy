import numpy as np
import scipy.special as sc

from sampy.distributions import Discrete
from sampy.interval import Interval
from sampy.utils import check_array
from sampy.math import logn, _handle_zeros_in_scale


class Binomial(Discrete):
	def __init__(self, n_trials=1, bias=0.5, seed=None):
		self.n_trials = n_trials
		self.bias = bias
		self.seed = seed
		self._state = self._set_random_state(seed)

	@classmethod
	def from_data(self, X, seed=None):
		dist = Binomial(seed=seed)
		return dist.fit(X)

	def fit(self, X):
		raise NotImplementedError

	def partial_fit(self, X):
		# check array for numpy structure
		X = check_array(X, squeeze=True).astype(float)

		# identify values outside of support
		invalid = (1 - self.support.contains(X)).astype(bool)
		X[invalid] = np.nan

		raise NotImplementedError

	def sample(self, *size):
		return self._state.binomial(self.n_trials, self.bias, size=size)

	def pmf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)
		return np.exp(self.log_pmf(X))

	def log_pmf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)
		
		# Floor values of X
		X = np.floor(X)

		# Expand all components of log-pmf
		out = sc.gammaln(self.n_trials + 1) - \
			(sc.gammaln(X + 1) + sc.gammaln(self.n_trials - X + 1)) + \
			sc.xlogy(X, self.bias) + sc.xlog1py(self.n_trials - X, -self.bias)
		return out

	def cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		# floor X values
		X = np.floor(X)
		
		return sc.bdtr(X, self.n_trials, self.bias)

	def log_cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)
		
		return np.log(self.cdf(X))

	def quantile(self, *q):
		# check array for numpy structure
		q = check_array(q, squeeze=True)

		vals = np.ceil(sc.bdtrik(q, self.n_trials, self.bias))
		vals1 = np.maximum(vals - 1, 0)
		temp = sc.bdtr(vals1, self.n_trials, self.bias)
		return np.where(temp >= q, vals1, vals).astype(int)

	@property
	def mean(self):
		return self.n_trials * self.bias

	@property
	def median(self):
		return self.quantile(0.5)

	@property
	def mode(self):
		return self.median

	@property
	def variance(self):
		return self.n_trials * self.bias * (1 - self.bias)

	@property
	def skewness(self):
		n, p, q = self.n_trials, self.bias, 1 - self.bias
		return (q - p) / np.sqrt(n * p * q)

	@property
	def kurtosis(self):
		n, p, q = self.n_trials, self.bias, 1 - self.bias
		return (1 - (6 * p * q)) / (n * p * q)

	def entropy(self):
		n, p, q = self.n_trials, self.bias, 1 - self.bias
		return 0.5 * logn(2 * np.pi * np.exp(1) * n * p * q, 2)

	def perplexity(self):
		return np.exp(self.entropy())

	@property
	def support(self):
		return Interval(0, self.n_trials, True, True)

	def _reset(self):
		if hasattr(self, '_n_samples'):
			del self._n_samples
		self.rate = None

	def __str__(self):
		return f"Binomial(n_trials={self.n_trials}, bias={self.bias})"

	def __repr__(self):
		return self.__str__()
