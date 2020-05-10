import numpy as np
import scipy.special as sc

from sampy.distributions import Continuous
from sampy.interval import Interval
from sampy.utils import check_array


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
		raise NotImplementedError()

	def fit(self, X):
		raise NotImplementedError()

	def partial_fit(self, X):
		raise NotImplementedError()

	def sample(self, *size):
		return self._state.gamma(self.shape, self.rate, size=size)

	def pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		return np.exp(self.log_pdf(X))

	def log_pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		# alias parameters
		a, b = self.shape, self.rate

		return a * np.log(b) + (a - 1) * np.log(X) - b * X - sc.gammaln(a)

	def cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		# alias parameters
		a, b = self.shape, self.rate

		return sc.gammainc(a, b * X) / sc.gamma(a)

	def log_cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		return np.log(self.cdf(X))

	def quantile(self, *q):
		# check array for numpy structure
		q = check_array(q, squeeze=True)

		# alias parameters
		a, b = self.shape, self.rate

		return sc.gammaincinv(a, b * q)

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

	@property
	def support(self):
		return Interval(0, np.inf, False, False)

	def __str__(self):
		return f"Gamma(shape={self.shape}, rate={self.rate})"

	def __repr__(self):
		return self.__str__()


