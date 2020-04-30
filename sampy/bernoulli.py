import numpy as np
import scipy.special as sc

from sampy.distributions import Discrete
from sampy.interval import Interval
from sampy.utils import check_array
from sampy.math import _handle_zeros_in_scale


class Bernoulli(Discrete):
	""" Bernoulli Distribution

	Also known as the coin-flip or yes-no distribution, this distribution 
	describes the likelihood of a single trial returning with a positive (heads
	or yes) value with a given probability (or bias). This is a specific case
	of the more general Binomial distribution.

	Parameters
	----------
	bias : float, optional
		The bias or the likelihood of a positive event, by default 0.5. 
		The values within this method assume that negative class will be 0
		and positive class 1. 
	seed : int, str or None, optional
		The seed by which to set the random number generator for the 
		`sample` method, by default None. A seed set to None will use the
		default clock time (see numpy.random.RandomState for more details)
		and an integer will set the seed directly. Strings will be hashed to an
		integer representation which can be a helpful description of the 
		distribution use or associated experiment. 
	"""

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

		# convert values outside of support

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

	def pmf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		out = np.zeros(X.shape)
		out[X == 0] = 1 - self.bias
		out[X == 1] = self.bias
		return out

	def log_pmf(self, *X):
		# check array for numpy structure
		X = check_array(X, squeeze=True)

		return np.log(self.pmf(X))

	def cdf(self, *X):
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

	def quantile(self, *q):
		# check array for numpy structure
		q = check_array(q, squeeze=True)

		out = np.ceil(sc.bdtrik(q, 1, self.bias))
		return np.where(self.bias >= q, 0, out)

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

	@property
	def support(self):
		return Interval(0, 1, True, True)

	def _reset(self):
		if hasattr(self, '_n_samples'):
			del self._n_samples
		self.bias = None

	def __str__(self):
		return f"Bernoulli(bias={self.bias})"

	def __repr__(self):
		return self.__str__()
