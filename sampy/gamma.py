import numpy as np
import scipy.special as sc

from sampy.distributions import Continuous
from sampy.interval import Interval
from sampy.utils import check_array, cache_property


class Gamma(Continuous):
	def __init__(self, shape=1, rate=1, seed=None):

		if shape <= 0:
			raise ValueError("Shape parameter must be greater than 0")

		if rate <= 0:
			raise ValueError("Rate parameter must be greater than 0")

		self.shape = shape
		self.rate = rate
		self.seed = seed
		self._state = self._set_random_state(seed)

	@classmethod
	def from_data(self, X, seed=None):
		dist = Gamma(seed=seed)
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

		# update mean 
		if self.shape is None and self.rate is None:
			self._n_samples += X.shape[0] - np.isnan(X).sum()
			self._mean = np.nanmean(X)
			self._log_mean = np.nanmean(np.log(X))
		else:
			# previous values
			prev_size = self._n_samples
			prev_mean = self._mean
			prev_log_mean = self._log_mean

			# current values
			curr_size = X.shape[0] - np.isnan(X).sum()
			curr_mean = np.nanmean(X)
			curr_log_mean = np.nanmean(np.log(X))

			# update size
			self._n_samples = prev_size + curr_size

			# update mean
			self._mean = ((prev_mean * prev_size) + \
				(curr_mean * curr_size)) / self._n_samples

			# update log-mean
			self._log_mean = ((prev_log_mean * prev_size) + \
				(curr_log_mean * curr_size)) / self._n_samples

		# solving for shape parameter has no analytical closed form solution
		# however shape is numerically well behaved and can be computed with 
		# some level of numerical stability. Below we estimate parameter `s`
		# which aids in the estimation of shape parameter `k`.
		s = np.log(self._mean) - self._log_mean
		k = (3 - s + np.sqrt((s - 3) * (s - 3) + 24 * s)) / (12 * s)

		# this estimation of k is within 1.5% of correct value updated with 
		# explicit form of Newton-Raphson
		k -= (np.log(k) - sc.psi(k) - s) / ((1 / k) - sc.psi(k))

		# solve for theta (theta = 1 / self.rate)
		theta = self._mean / k
		
		# update parameters
		self.shape = k
		self.rate = 1 / theta
		return self

	def sample(self, *size):
		return self._state.gamma(self.shape, 1 / self.rate, size=size)

	def pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		return np.exp(self.log_pdf(X))

	def log_pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		# alias parameters
		a, b = self.shape, self.rate

		return a * np.log(b) + (a - 1) * np.log(X) - b * X - sc.gammaln(a)

	def cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		# alias parameters
		a, b = self.shape, self.rate

		return sc.gammainc(a, b * X)

	def log_cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		return np.log(self.cdf(X))

	def quantile(self, *q):
		# check array for numpy structure
		q = check_array(q, reduce_args=True, ensure_1d=True)

		# alias parameters
		a, b = self.shape, self.rate

		return sc.gammaincinv(a, q) / b

	@property
	def mean(self):
		return self.shape / self.rate

	@property
	def median(self):
		return self.quantile(0.5)

	@property
	def mode(self):
		"""No closed form exist only valid for shape >= 1"""
		return (self.shape - 1) / self.rate

	@property
	def variance(self):
		return self.shape / (self.rate ** 2)

	@property 
	def skewness(self):
		return 2 / np.sqrt(self.shape)

	@property
	def kurtosis(self):
		return 6 / self.shape

	@property
	def entropy(self):
		# alias parameters
		a, b = self.shape, self.rate

		return a - np.log(b) - sc.gammaln(a) + (1 - a) * sc.psi(a)

	@property
	def perplexity(self):
		return np.exp(self.entropy)

	@cache_property
	def support(self):
		return Interval(0, np.inf, False, False)

	def _reset(self):
		if hasattr(self, '_n_samples'):
			del self._n_samples
		self.shape = None
		self.rate = None

	def __str__(self):
		return f"Gamma(shape={self.shape}, rate={self.rate})"

	def __repr__(self):
		return self.__str__()


