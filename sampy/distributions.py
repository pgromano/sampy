import numpy as np
import sys


class Distribution:
	""" Base Distribution Class """

	def fit(self, X):
		raise NotImplementedError

	@property
	def mean(self):
		raise NotImplementedError

	@property
	def median(self):
		raise NotImplementedError

	@property
	def mode(self):
		raise NotImplementedError

	@property
	def variance(self):
		raise NotImplementedError

	@property
	def skewness(self):
		raise NotImplementedError

	@property
	def kurtosis(self):
		raise NotImplementedError

	def entropy(self):
		raise NotImplementedError

	def perplexity(self):
		raise NotImplementedError

	def sample(self, *size):
		if not hassattr(self, '_state'):
			raise ValueError("No random state found in method")

		if not hasattr(self, 'icdf'):
			raise ValueError("Must define inverse CDF function")

		p = self._state.uniform(0, 1, size=size)
		return self.quantile(p)

	def _set_random_state(self, seed):
		if isinstance(seed, np.random.RandomState):
			return seed

		if isinstance(seed, str):
			seed = hash(seed) & ((1 << 32) - 1)

		return np.random.RandomState(seed)

	def __call__(self, *size):
		return self.sample(*size)


class Discrete(Distribution):
	""" Base Discrete Distribution Class """
	def pmf(self, X):
		raise NotImplementedError

	def log_pmf(self, X):
		raise NotImplementedError

	def cdf(self, X):
		raise NotImplementedError

	def log_cdf(self, X):
		raise NotImplementedError

	def quantile(self, q):
		raise NotImplementedError

class Continuous(Distribution):
	""" Base Continuous Distribution Class """

	def pdf(self, X):
		raise NotImplementedError

	def log_pdf(self, X):
		raise NotImplementedError

	def cdf(self, X):
		raise NotImplementedError

	def log_cdf(self, X):
		raise NotImplementedError

	def quantile(self, q):
		raise NotImplementedError
