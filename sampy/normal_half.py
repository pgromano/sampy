import numpy as np
import scipy.special as sc

from sampy.distributions import Continuous
from sampy.interval import Interval
from sampy.utils import check_array
from sampy.math import _handle_zeros_in_scale, logn


class HalfNormal(Continuous):
	def __init__(self, scale=1, seed=None):
		self.scale = scale
		self.seed = seed
		self._state = self._set_random_state(seed)

	@classmethod
	def from_data(self, X, seed=None):
		dist = HalfNormal(seed=seed)
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

		# Update center and variance
		if self._empirical_variance is None:
			self._n_samples += X.shape[0] - np.isnan(X).sum()
			self._empirical_variance = np.nanvar(X)
		else:
			# previous values
			prev_size = self._n_samples
			prev_variance = self._empirical_variance

			# new values
			curr_size = X.shape[0] - np.isnan(X).sum()
			curr_variance = np.nanvar(X)

			# update size
			self._n_samples = prev_size + curr_size

			# update variance
			self._empirical_variance = ((prev_variance * prev_size) +
                            (curr_variance * curr_size)) / self._n_samples

		norm = (1 - (2 / np.pi))
		self.scale = _handle_zeros_in_scale(
			np.sqrt(self._empirical_variance / norm)
		)
		return self

	def pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		norm = np.sqrt(2) / (self.scale * np.sqrt(np.pi))
		p = norm * np.exp(-X ** 2 / (2 * self.scale ** 2))
		return np.where(X > 0, p, 0)

	def log_pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		norm = np.log(np.sqrt(2)) - np.log(self.scale * np.sqrt(np.pi))
		p = norm - (X ** 2 / (2 * self.variance))
		return np.where(X >= 0, p, 1)

	def cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		return sc.erf(X / (np.sqrt(2) * self.scale))

	def log_cdf(self, *X):

		return np.log(self.cdf(X))

	def quantile(self, *q):
		# check array for numpy structure
		q = check_array(q, reduce_args=True, ensure_1d=True)

		return self.scale * np.sqrt(2) * sc.erfinv(q)

	@property
	def mean(self):
		return (self.scale * np.sqrt(2)) / np.sqrt(np.pi)

	@property
	def median(self):
		return self.scale * np.sqrt(2) * sc.erfinv(0.5)

	@property
	def mode(self):
		return 0

	@property
	def variance(self):
		return (self.scale ** 2) * (1 - (2 / np.pi))

	@property
	def skewness(self):
		return (np.sqrt(2) * (4 - np.pi)) / np.power(np.pi - 2, 3 / 2)

	@property
	def kurtosis(self):
		return (8 * (np.pi - 3)) / np.pow(np.pi - 2, 2)

	@property
	def entropy(self):
		return 0.5 * np.log((np.pi * self.scale ** 2) / 2) + 0.5

	@property
	def perplexity(self):
		return np.exp(self.entropy)

	@property
	def support(self):
		return Interval(0, np.inf, True, False)

	def _reset(self):
		if hasattr(self, '_n_samples'):
			del self._n_samples
		self.scale = None
		self._empirical_variance = None

	def __str__(self):
		return f"HalfNormal(scale={self.scale})"

	def __repr__(self):
		return self.__str__()
