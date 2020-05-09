import numpy as np

from sampy.distributions import Discrete
from sampy.interval import Interval
from sampy.utils import check_array


class DiscreteUniform(Discrete):
	def __init__(self, low=0, high=1, upper_inclusive=True, seed=None):

		if upper_inclusive:
			if low > high:
				err = f"High must be greater than low: {low} ≮ {high}"
				raise ValueError(err)
			elif low == high:
				err = f"Lower and upper bounds cannot be equal: {low} = {high}"
				raise ValueError(err)
		else:
			caveat = "\nHalf interval distribution (upper_inclusive = False) "
			caveat += "shifts upper bound (high - 1) in comparison to low."
			if low > high - 1:
				err = f"High must be greater than low: {low} ≮ {high} - 1"
				raise ValueError(err + caveat)
			elif low == high - 1:
				err = f"Lower and upper bounds cannot be equal: {low} = {high} - 1"
				raise ValueError(err + caveat)

		self.low = low
		self.high = high
		self.upper_inclusive = upper_inclusive
		self.seed = seed
		self._state = self._set_random_state(seed)

	@classmethod
	def from_data(self, X, upper_inclusive=False, seed=None):
		dist = DiscreteUniform(upper_inclusive=upper_inclusive, seed=seed)
		return dist.fit(X)

	def fit(self, X):
		self._reset()
		return self.partial_fit(X)

	def partial_fit(self, X):

		# check array for numpy structure
		X = check_array(X, squeeze=True, dtype=int)

		# First fit
		if self.low is None and self.high is None:
			self.low = np.nanmin(X)
			self.high = np.nanmax(X) + (1 - self.upper_inclusive)
		else:
			# Update distribution support
			curr_low, curr_high = np.nanmin(X), np.nanmax(X)
			if curr_low < self.low:
				self.low = curr_low

			if curr_high > self.high:
				self.high = curr_high + (1 - self.upper_inclusive)

		return self

	def sample(self, *size):
		if self.upper_inclusive: 
			return self._state.randint(self.low, self.high + 1, size=size)
		return self._state.randint(self.low, self.high, size=size)

	def pmf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True, dtype=int)

		lb = self.low <= X
		if self.upper_inclusive:
			ub = self.high >= X
		else:
			ub = self.high > X

		if self.upper_inclusive:
			nrange = self.high - self.low
		else:
			nrange = (self.high - 1) - self.low
		return (lb * ub) / nrange

	def log_pmf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True, dtype=int)

		lb = self.low <= X 
		if self.upper_inclusive:
			ub = self.high >= X
			nrange = self.high - self.low
		else:
			ub = self.high > X
			nrange = (self.high - 1) - self.low

		return np.log(lb * ub) - np.log(nrange)

	def cdf(self, *X):
		# check array for numpy structure
		X = np.floor(check_array(X, squeeze=True, dtype=int))

		if self.upper_inclusive:
			return np.clip((X - self.low) / (self.high - self.low), 0, 1)
		return np.clip((X - self.low) / ((self.high - 1) - self.low), 0, 1)

	def log_cdf(self, *X):

		return np.log(self.cdf(X))

	def quantile(self, *q):
		# check array for numpy structure
		q = check_array(q, squeeze=True)

		if self.upper_inclusive:
			return self.low + q * (self.high - self.low)
		return self.low + q * ((self.high - 1) - self.low)

	@property
	def mean(self):
		if self.upper_inclusive:
			return 0.5 * (self.high - self.low) + self.low
		return 0.5 * ((self.high - 1) - self.low) + self.low

	@property
	def median(self):
		return np.round(self.mean)

	@property
	def mode(self):
		return np.nan

	@property
	def variance(self):
		if self.upper_inclusive:
			return (self.high - self.low) ** 2 / 12
		return ((self.high - 1) - self.low) ** 2 / 12

	@property
	def skewness(self):
		return 0

	@property
	def kurtosis(self):
		return -6 / 5

	@property
	def entropy(self):
		if self.upper_inclusive:
			return np.log(self.high - self.low)
		return np.log((self.high - 1) - self.low)

	@property
	def perplexity(self):
		return np.exp(self.entropy)

	@property
	def support(self):
		if self.upper_inclusive:
			return Interval(self.low, self.high, True, True)
		return Interval(self.low, self.high, True, False)

	def _reset(self):
		self.low = None
		self.high = None

	def __str__(self):
		return f"DiscreteUniform(low={self.low}, high={self.high - (1 - self.upper_inclusive)})"

	def __repr__(self):
		return self.__str__()
