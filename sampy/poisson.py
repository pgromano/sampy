import numpy as np
import scipy.special as sc

from sampy.distributions import Discrete
from sampy.utils import check_array
from sampy.math import _handle_zeros_in_scale


class Poisson(Discrete):
	def __init__(self, rate=1, seed=None):
		self.rate = rate
		self.seed = seed
		self._state = self._set_random_state(seed)

	@classmethod
	def from_data(self, X, seed=None):
		dist = Poisson(seed=seed)
		return dist.fit(X)

	def fit(self, X):
		self._reset()
		return self.partial_fit(X)

	def partial_fit(self, X):

		# check array for numpy structure
		X = check_array(X, squeeze=True)

		# first fit
		if not hasattr(self, '_n_samples'):
			self._n_samples = 0

		# Update rate
		if self.rate is None:
			self._n_samples += X.shape[0] - np.isnan(X).sum()
			self.rate = np.nanmean(X)
		else:
			# previous values
			prev_size = self._n_samples
			prev_rate = self.rate

			# new values
			curr_size = X.shape[0] - np.isnan(X).sum()
			curr_rate = np.nanmean(X)

			# update size
			self._n_samples = prev_size + curr_size

			# update rate
			self.rate = ((prev_rate * prev_size) +
				(curr_rate * curr_size)) / self._n_samples

		return self

	def sample(self, *size):
		return self._state.poisson(self.rate, size=size)

	def pmf(self, *X):

		return np.exp(self.log_pmf(X))

	def log_pmf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		return (np.log(self.rate) * X) - self.rate - sc.gammaln(X + 1)

	def cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)
		
		return sc.pdtr(np.floor(X), self.rate)

	def log_cdf(self, *X):
		return np.log(self.cdf(X))

	def quantile(self, *q):
		# check array for numpy structure
		q = check_array(q, squeeze=True)

		vals = np.ceil(sc.pdtrik(q, self.rate))
		vals1 = np.maximum(vals - 1, 0)
		temp = sc.pdtr(vals1, self.rate)
		return np.where(temp >= q, vals1, vals)

	@property
	def mean(self):
		return self.rate

	@property
	def median(self):
		return np.floor(self.rate + (1 / 3) - 0.02 / self.rate)

	@property
	def mode(self):
		return np.floor(self.rate)

	@property
	def variance(self):
		return self.rate

	@property
	def skewness(self):
		return 1 / np.sqrt(self.rate)

	@property
	def kurtosis(self):
		return 1 / self.rate

	@property
	def entropy(self):
		return np.nan

	@property
	def perplexity(self):
		return np.nan

	@property
	def support(self):
		return Interval(0, np.inf, True, False)

	def _reset(self):
		if hasattr(self, '_n_samples'):
			del self._n_samples
		self.rate = None

	def __str__(self):
		return f"Poisson(rate={self.rate})"

	def __repr__(self):
		return self.__str__()
