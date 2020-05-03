import numpy as np
import scipy.special as sc

from sampy.distributions import Continuous
from sampy.interval import Interval
from sampy.utils import check_array
from sampy.math import _handle_zeros_in_scale, logn

class Normal(Continuous):
	def __init__(self, center=0, scale=1, seed=None):
		self.center = center
		self.scale = scale
		self._variance = scale ** 2
		self.seed = seed
		self._state = self._set_random_state(seed)

	@classmethod
	def from_data(self, X, seed=None):
		dist = Normal(seed=seed)
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

		# Update center and variance
		if self.center is None and self._variance is None:
			self._n_samples += X.shape[0] - np.isnan(X).sum()
			self.center = np.nanmean(X)
			self._variance = np.nanvar(X)
		else:
			# previous values
			prev_size = self._n_samples
			prev_center = self.center
			prev_variance = self._variance

			# new values
			curr_size = X.shape[0] - np.isnan(X).sum()
			curr_center = np.nanmean(X)
			curr_variance = np.nanvar(X)

			# update size
			self._n_samples = prev_size + curr_size

			# update center
			self.center = ((prev_center * prev_size) + \
				(curr_center * curr_size)) / self._n_samples

			# update variance
			self._variance = ((prev_variance * prev_size) + \
				(curr_variance * curr_size)) / self._n_samples

		self.scale = _handle_zeros_in_scale(np.sqrt(self.variance))
		return self

	def sample(self, *size):
		return self._state.normal(self.center, self.scale, size=size)

	def pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		norm = (self.scale * np.sqrt(2 * np.pi))
		return np.exp(-0.5 * ((X - self.center) / self.scale) ** 2) / norm

	def log_pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		norm = 2 * self.variance
		log_scale = np.log(self.scale) + np.log(np.sqrt(2 * np.pi))
		return -((X - self.center) ** 2) / norm - log_scale

	def cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		return 0.5 * (1 + sc.erf((X - self.center) / (np.sqrt(2) * self.scale)))

	def log_cdf(self, *X):

		return np.log(self.cdf(X))

	def quantile(self, *q):
		# check array for numpy structure
		q = check_array(q, squeeze=True)

		return self.center + self.scale * sc.erfinv(2 * q - 1) * np.sqrt(2)

	@property
	def mean(self):
		return self.center

	@property
	def median(self):
		return self.center

	@property
	def mode(self):
		return self.median

	@property
	def variance(self):
		return self._variance

	@property
	def skewness(self):
		return 0

	@property
	def kurtosis(self):
		return 0

	def entropy(self):
		return 0.5 + 0.5 * np.log(2 * np.pi) + np.log(self.scale)

	def perplexity(self):
		return np.exp(self.entropy)

	@property
	def support(self):
		return Interval(-np.inf, np.inf, False, False)

	def _reset(self):
		if hasattr(self, '_n_samples'):
			del self._n_samples
		self.center = None
		self.scale = None
		self._variance = None

	def __str__(self):
		return f"Normal(center={self.center}, scale={self.scale})"

	def __repr__(self):
		return self.__str__()
