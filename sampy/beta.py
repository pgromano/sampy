import warnings
import numpy as np
import scipy.special as sc
import scipy.optimize as opt

from sampy.distributions import Continuous
from sampy.interval import Interval
from sampy.utils import check_array, cache_property


def _log_loss(params, mu, nu, n):
	# Expand beta parameters
	a, b = params

	# Functions for root solutions
	func1 = n * mu - n * (-sc.psi(a + b) + sc.psi(a))
	func2 = n * nu - n * (-sc.psi(a + b) + sc.psi(b))
	return [func1, func2]


class Beta(Continuous):
	def __init__(self, alpha=1, beta=1, fit_raise='error', seed=None):
		self.alpha = alpha
		self.beta = beta
		self.seed = seed
		self.fit_raise = fit_raise
		self._state = self._set_random_state(seed)

	@classmethod
	def from_data(self, X, seed=None):
		dist = Beta(seed=seed)
		return dist.fit(X)

	def fit(self, X):
		self._reset()
		return self.partial_fit(X)

	def partial_fit(self, X):

		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		# check domain
		if (1 - self.support.contains(X).sum() > 0):
			raise ValueError("Training data not within support domain")

		# first fit
		if not hasattr(self, '_n_samples'):
			self._n_samples = 0

		# Update center and variance
		if self._mean is None:
			self._n_samples += X.shape[0] - np.isnan(X).sum()
			self._mean = np.nanmean(X)
			self._mu = np.nanmean(np.log(X))
			self._nu = np.nanmean(np.log1p(-X))
			self._variance = np.nanvar(X)
		else:
			# previous values
			prev_size = self._n_samples
			prev_mean = self._mean
			prev_variance = self._variance
			prev_mu = self._mu
			prev_nu = self._nu

			# new values
			curr_size = X.shape[0] - np.isnan(X).sum()
			curr_mean = np.nanmean(X)
			curr_variance = np.nanvar(X)
			curr_mu = np.nanmean(np.log(X))
			curr_nu = np.nanmean(np.log1p(-X))

			# update size
			self._n_samples = prev_size + curr_size

			# update mean
			self._mean = ((prev_mean * prev_size) +
						(curr_mean * curr_size)) / self._n_samples

			# update variance
			self._variance = ((prev_variance * prev_size) +
							(curr_variance * curr_size)) / self._n_samples
			# update mu
			self._mu = ((prev_mu * prev_size) +
						(curr_mu * curr_size)) / self._n_samples

			# update nu
			self._nu = ((prev_nu * prev_size) +
						(curr_nu * curr_size)) / self._n_samples

		m, v = self._mean, self._variance
		# m, v = X.mean(), X.var(ddof=0)
		norm = m * (1 - m) / v - 1
		alpha = m * norm
		beta = (1 - m) * norm

		params, info, ier, msg = opt.fsolve(
			_log_loss, [alpha, beta],
			args=(self._mu, self._nu, self._n_samples),
			full_output=True
		)

		if ier != 1:
			if self.fit_raise == 'error':
				raise ValueError("Fit failed:\n\n" + msg)
			elif self.fit_raise == 'warn':
				if self.alpha is None and self.beta is None:
					warnings.warn(
						f"Unable to optimize beta log-loss – parameters will be from method of moments optimization: alpha={alpha}, beta={beta}.\n\nMLE optimization output:\n\n" + msg)
				else:
					warnings.warn(
						f"Unable to optimize beta log-loss – parameters will not update: alpha={alpha}, beta={beta}.\n\nMLE optimization output:\n\n" + msg)
			self.alpha, self.beta = alpha, beta
		else:
			self.alpha, self.beta = params

		return self

	def sample(self, *size):
		return self._state.beta(self.alpha, self.beta, size=size)

	def pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		norm = sc.beta(self.alpha, self.beta)
		p = np.power(X, self.alpha - 1) * np.power(1 - X, self.beta - 1)
		return p / norm

	def log_pdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		norm = sc.betaln(self.alpha, self.beta)
		p = (self.alpha - 1) * np.log(X) + (self.beta - 1) * np.log(1 - X)
		return p - norm

	def cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)

		return sc.btdtr(self.alpha, self.beta, X)

	def log_cdf(self, *X):
		# check array for numpy structure
		X = check_array(X, reduce_args=True, ensure_1d=True)
		
		return np.log(self.cdf(X))

	def quantile(self, *q):
		# check array for numpy structure
		q = check_array(q, reduce_args=True, ensure_1d=True)
		
		return sc.btdtri(self.alpha, self.beta, q)

	@property
	def mean(self):
		return self.alpha / (self.alpha + self.beta)

	@property
	def median(self):
		return self.quantile(0.5)

	@property
	def mode(self):
		a, b = self.alpha, self.beta
		if a < 1 and b < 1:
			return np.nan
		elif a <= 1 and b > 1:
			return 0
		elif a > 1 and b <= 1:
			return 1
		return (a - 1) / (a + b - 2)

	@property
	def variance(self):
		a, b = self.alpha, self.beta
		return (a * b) / ((a + b) * (a + b) * (a + b + 1))

	@property
	def skewness(self):
		a, b = self.alpha, self.beta

		norm = (a + b + 2) * np.sqrt(a * b)
		out = 2 * (b - a) * np.sqrt(a + b + 1)
		return out / norm

	@property
	def kurtosis(self):
		a, b = self.alpha, self.beta
		norm = a * b * (a + b + 2) * (a + b + 3)
		out = 6 * [(a + b) * (a + b) * (a + b + 1) - a * b * (a + b + 2)]
		return out / norm

	@property
	def entropy(self):
		a, b = self.alpha, self.beta
		out = sc.betaln(a, b) - (a - 1) * sc.digamma(a)
		out -= (b - 1) * sc.digamma(b)
		out += (a + b - 2) * sc.digamma(a + b)
		return out

	@property
	def perplexity(self):
		return np.exp(self.entropy)

	@cache_property
	def support(self):
		return Interval(0, 1, True, True)

	def _reset(self):
		if hasattr(self, '_n_samples'):
			del self._n_samples
		self.alpha = None
		self.beta = None

		# For MLE estimation initial estimate is made from empirical
		# mean and variance and then optimized by root finding the
		# alpha and beta parameters from the two-parameter log-loss
		self._mean = None
		self._mu = None
		self._nu = None
		self._variance = None

	def __str__(self):
		return f"Beta(alpha={self.alpha}, beta={self.beta})"

	def __repr__(self):
		return self.__str__()


