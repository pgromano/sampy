import numpy as np
import scipy.special as sc
import warnings

from sampy.interval import Interval
from sampy.utils import check_array, cache_property


class Multinomial:
	def __init__(self, n_trials=1, bias=[0.5, 0.5], seed=None):
		
		# safety check n_trials
		if isinstance(n_trials, float):
			warnings.warn("Number of trials should be an integer. Casting to int")

		if np.round(n_trials).astype(int) != n_trials:
			raise ValueError("Number of trials must be an integer value")
		else:
			n_trials = np.round(n_trials).astype(int)

		# safety check bias
		bias = check_array(bias, ensure_1d=True)
		if any(bias < 0) or any(bias > 1):
			raise ValueError("Bias must be between [0, 1]")

		self.n_trials = n_trials
		self.bias = bias
		self.seed = seed
		self._state = self._set_random_state(seed)

	def sample(self, *size):
		return self._state.multinomial(self.n_trials, self.bias)

	def pmf(self, *X):

		# check array for numpy structure
		# NOTE: feature_axis set to rows to ensure that *args  that represent a
		# single observtion will be the correct shape. Otherwise, users will
		# *have* to pass correct shape for multiple observations (which does not
		# effect the final shape)
		X = check_array(X, reduce_args=True, atleast_2d=True, feature_axis=0)

		return np.exp(self.log_pmf(X))

	def log_pmf(self, *X):
		""" Log Probability Mass Function

		Parameters
		----------
		X : int or array-like
		"""

		# check array for numpy structure
		X = check_array(X, reduce_args=True, atleast_2d=True, feature_axis=0)

		# compute mass
		log_mass = sc.gammaln(self.n_trials + 1) + \
			np.sum(sc.xlogy(X, self.bias) - sc.gammaln(X + 1), axis=1)

		# check for invalid values
		out_bounds = np.any(self.support.not_contains(X), axis=1)
		invalid_input = X.sum(1) != self.n_trials
		log_mass[np.logical_or(out_bounds, invalid_input)] = np.nan

		return log_mass

	@property
	def mean(self):
		return self.n_trials * self.bias

	@property
	def variance(self):
		return self.n_trials * self.bias * (1 - self.bias)

	@property
	def covariance(self):
		return -self.n_trials * np.outer(self.bias, self.bias)

	@property
	def entropy(self):
		raise NotImplementedError

	@property
	def perplexity(self):
		raise NotImplementedError

	@cache_property
	def support(self):
		return Interval(0, self.n_trials, True, True)

	def _set_random_state(self, seed):
		return set_random_state(seed)

	def __call__(self, *size):
		return self.sample(*size)

	def _reset(self):
		self.n_trials = None
		self.bias = None

	def __str__(self):
		return f"Multinomial(n_trials={self.n_trials}, bias={self.bias})"

	def __repr__(self):
		return self.__str__()
