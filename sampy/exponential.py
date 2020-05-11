import numpy as np
import scipy.special as sc

from sampy.distributions import Continuous
from sampy.interval import Interval
from sampy.utils import check_array
from sampy.math import _handle_zeros_in_scale, logn


class Exponential(Continuous):
	def __init__(self, rate=1, seed=None):
		self.rate = rate
		self.seed = seed
		self._state = self._set_random_state(seed)

	@classmethod
	def from_data(self, X, seed=None):
		dist = Exponential(seed=seed)
		return dist.fit(X)

	def fit(self, X):
		self._reset()
		return self.partial_fit(X)

	def partial_fit(self, X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		# first fit
		if not hasattr(self, '_n_samples'):
			self._n_samples = 0

		# Update rate
		if self.rate is None:
			self._n_samples += X.shape[0] - np.isnan(X).sum()
			self.rate = 1 / np.nanmean(X)
		else:
			# previous values
			prev_size = self._n_samples
			prev_mean = 1 / self.rate

			# new values
			curr_size = X.shape[0] - np.isnan(X).sum()
			curr_mean = np.nanmean(X)

			# update size
			self._n_samples = prev_size + curr_size

			# update rate
			updated_mean = ((prev_mean * prev_size) + \
				(curr_mean * curr_size)) / self._n_samples
			self.rate = 1 / updated_mean

		return self

	def sample(self, *size):
		return self._state.exponential(1 / self.rate, size=size)
	
	def pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		return self.rate * np.exp(-self.rate * X)

	def log_pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze = True)

		return np.log(self.rate) - self.rate * X

	def cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		return 1 - np.exp(-self.rate * X)

	def log_cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		return self.rate * X

	def quantile(self, *q):
		# check array for numpy structure
		q = check_array(q, reduce_args=True, ensure_1d=True)

		return -np.log(1 - q) / self.rate

	@property
	def mean(self):
		return 1 / self.rate

	@property
	def median(self):
		return np.log(2) / self.rate

	@property
	def mode(self):
		return 0

	@property
	def variance(self):
		return 1 / (self.rate ** 2)

	@property
	def skewness(self):
		return 2

	@property
	def kurtosis(self):
		return 6

	@property
	def entropy(self):
		return 1 - np.log(self.rate)

	@property
	def perplexity(self):
		return np.exp(self.entropy)

	@property
	def support(self):
		return Interval(0, np.inf, True, False)

	def _reset(self):
		if hasattr(self, '_n_samples'):
			del self._n_samples
		self.rate = None

	def __str__(self):
		return f"Exponential(rate={self.rate})"

	def __repr__(self):
		return self.__str__()
