import numpy as np
import scipy.special as sc

from sampy.distributions import Discrete
from sampy.utils import check_array
from sampy.math import _handle_zeros_in_scale


class Bernoulli(Discrete):
	def __init__(self, bias=0.5, seed=None):
		self.bias = bias
		self.seed = seed
		self._state = self._set_random_state(seed)

	@classmethod
	def from_data(self, X, seed=None):
		dist = Bernoulli(seed=seed)
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
		if self.bias is None:
			self._n_samples += X.shape[0] - np.isnan(X).sum()
			self.bias = np.nanmean(X)
		else:
			# previous values
			prev_size = self._n_samples
			prev_rate = self.bias

			# new values
			curr_size = X.shape[0] - np.isnan(X).sum()
			curr_rate = np.nanmean(X)

			# update size
			self._n_samples = prev_size + curr_size

			# update rate
			self.bias = ((prev_rate * prev_size) +
				(curr_rate * curr_size)) / self._n_samples

		return self

	def sample(self, *size):
		return self._state.binomial(1, self.bias, size=size)

	def pmf(self, X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		out = np.zeros(X.shape)
		out[X == 0] = 1 - self.bias
		out[X == 1] = self.bias
		return out

	def log_pmf(self, X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		return np.log(self.pmf(X))

	def cdf(self, X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		out = np.zeros(X.shape)
		out[np.logical_or(X == 0, X == 1)] = 1 - self.bias
		out[X > 1] = 1
		return out 

	def log_cdf(self, X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		return np.log(self.cdf(X))

	def icdf(self, X):
		raise NotImplementedError

	def quantile(self, q):
		raise NotImplementedError

	@property
	def mean(self):
		return self.bias

	@property
	def median(self):
		if self.bias == 0.5:
			return np.nan
		return np.round(self.bias)
		
	@property
	def mode(self):
		if self.bias == 0.5:
			return np.nan
		return np.round(self.bias)

	@property
	def variance(self):
		return self.bias * (1 - self.bias)

	@property
	def skewness(self):
		p, q = self.bias, 1 - self.bias
		return (q - p) / np.sqrt(p * q)

	@property
	def kurtosis(self):
		p, q = self.bias, 1 - self.bias
		return (1 - 6 * p * q) / (p * q)

	def entropy(self):
		p, q = self.bias, 1 - self.bias
		return -q * np.log(q) - p * np.log(p)

	def perplexity(self):
		return np.exp(self.entropy())

	def _reset(self):
		if hasattr(self, '_n_samples'):
			del self._n_samples
		self.bias = None

	def __str__(self):
		return f"Bernoulli(bias={self.bias})"

	def __repr__(self):
		return self.__str__()
