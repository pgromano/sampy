import numpy as np

from sampy.distributions import Continuous
from sampy.utils import check_array


# class DiscreteUniform(Discrete):
# 	def __init__(self, low=0, high=1, seed=None):
# 		self.low = low
# 		self.high = high
# 		self._center = 0.5 * (self.high - self.low) + self.low
# 		self.seed = seed
# 		self._state = self._set_random_state(seed)


class Uniform(Continuous):
	def __init__(self, low=0, high=1, seed=None):
		self.low = low
		self.high = high
		self._center = 0.5 * (self.high - self.low) + self.low
		self.seed = seed
		self._state = self._set_random_state(seed)

	@classmethod
	def from_data(self, X, seed=None):
		dist = Uniform(seed=seed)
		return dist.fit(X)

	def fit(self, X):
		self._reset()
		return self.partial_fit(X)

	def partial_fit(self, X):

		# check array for numpy structure
		X = check_array(X, squeeze=True)

		# First fit
		if self.low is None and self.high is None:
			self.low = np.nanmin(X)
			self.high = np.nanmax(X)
		else:
			# Update distribution support
			curr_low, curr_high = np.nanmin(X), np.nanmax(X)
			if curr_low < self.low:
				self.low = curr_low

			if curr_high > self.high:
				self.high = curr_high

		self._center = 0.5 * (self.high - self.low) + self.low
		return self

	def sample(self, *size):
		return self._state.uniform(self.low, self.high, size=size)

	def pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		lb = self.low <= X
		ub = self.high > X
		return (lb * ub) / (self.high - self.low)

	def log_pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		lb = self.low <= X 
		ub = self.high > X
		return np.log(lb * ub) - np.log(self.high - self.low)

	def cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		return np.clip((X - self.low) / (self.high - self.low), 0, 1)

	def log_cdf(self, *X):

		return np.log(self.cdf(X))

	def quantile(self, *q):
		# check array for numpy structure
		q = check_array(q, squeeze=True)

		return self.low + q * (self.high - self.low)

	def entropy(self):
		return np.log(self.high - self.low)

	def perplexity(self):
		return np.exp(self.entropy())

	@property
	def mean(self):
		return self._center

	@property
	def median(self):
		return self._center

	@property
	def mode(self):
		return np.nan

	@property
	def variance(self):
		return (self.high - self.low) ** 2 / 12

	@property
	def skewness(self):
		return 0

	@property
	def kurtosis(self):
		return -6 / 5

	def _reset(self):
		if hasattr(self, '_center'):
			del self._center
		self.low = None
		self.high = None

	def __str__(self):
		return f"Uniform(low={self.low}, high={self.high})"

	def __repr__(self):
		return self.__str__()
