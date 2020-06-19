import numpy as np
import scipy.special as sc
import warnings

from sampy.distributions import Discrete
from sampy.interval import Interval
from sampy.utils import check_array, is_scalar, cache_property
from sampy.math import nanlog, logn


class Binomial(Discrete):
    """ Binomial Distribution

	This distribution describes the likelihood of returning a number of positive
    results (e.g. heads, yes, etc.)  from a fixed number of trials with a given 
    probability (or bias) of an individual positive event.

	Parameters
	----------
    n_trials : int, optional
        The number of independent Bernoulli trials with boolean outcome.
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
    def __init__(self, n_trials=1, bias=0.5, seed=None):

        # safety check n_trials
        if isinstance(n_trials, float):
            warnings.warn("Number of trials should be an integer. Casting to int")

        if np.round(n_trials).astype(int) != n_trials:
            raise ValueError("Number of trials must be an integer value")
        else:
            n_trials = np.round(n_trials).astype(int)

        # safety check bias
        if bias < 0 or bias > 1:
            raise ValueError("Bias must be between [0, 1]")

        self.n_trials = n_trials
        self.bias = bias
        self.seed = seed
        self._state = self._set_random_state(seed)

    @classmethod
    def from_data(self, X, seed=None):
        """Initialize distribution from data

        Parameters
        ----------
        X : numpy.ndarray
            1D dataset which falls within the domain of the given distribution
            support. The Binomial distribution expects positive integers only.
        seed : int, str or None, optional
            The seed by which to set the random number generator for the 
            `sample` method, by default None. A seed set to None will use the
            default clock time (see numpy.random.RandomState for more details)
            and an integer will set the seed directly. Strings will be hashed to 
            an integer representation which can be a helpful description of the 
            distribution use or associated experiment. 

        Returns
        -------
        sampy.Binomial
            The fitted Binomial distribution model 
        """
        dist = Binomial(seed=seed)
        return dist.fit(X)

    def fit(self, X):
        """Fit model to data

        This method fits the paramters for number of trials and bias by using
        method of moments estimation. The mean and variance of the provided data 
        are used to evaluated the most likely parameters.

        Parameters
        ----------
        X : numpy.ndarray
            1D dataset which falls within the domain of the given distribution
            support. The Binomial distribution expects positive integers only.

        Returns
        -------
        sampy.Binomial
            The fitted Binomial distribution model
        """
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """Incremental fit on a batch of samples

        This method partially fits (i.e. updates previous fits to new batch 
        data) the paramters for number of trials and bias by using method of 
        moments estimation. The mean and variance of the provided data are used 
        to evaluated the most likely parameters.

        Parameters
        ----------
        X : numpy.ndarray
            1D dataset which falls within the domain of the given distribution
            support. The Binomial distribution expects positive integers only.

        Returns
        -------
        sampy.Binomial
            The fitted Binomial distribution model
        """
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True).astype(float)

        # identify values outside of support
        # NOTE: we don't know the "true" upper bounds so we only
        # check that values are positive
        invalid = X < 0
        X[invalid] = np.nan

        # first fit
        if not hasattr(self, "_n_samples"):
            self._n_samples = 0

        if self._mean is None and self._variance is None:
            self._n_samples += X.shape[0] - np.isnan(X).sum()
            self._mean = np.nanmean(X)
            self._variance = np.nanvar(X)
        else:
            # previous values
            prev_size = self._n_samples
            prev_mean = self._mean
            prev_variance = self._variance

            # new values
            curr_size = X.shape[0] - np.isnan(X).sum()
            curr_mean = np.nanmean(X)
            curr_variance = np.nanvar(X)

            # update size
            self._n_samples = prev_size + curr_size

            # update mean
            self._mean = (
                (prev_mean * prev_size) + (curr_mean * curr_size)
            ) / self._n_samples

            # update variance
            self._variance = (
                (prev_variance * prev_size) + (curr_variance * curr_size)
            ) / self._n_samples

        self.bias = 1 - (self._variance / self._mean)
        self.n_trials = np.round(self._mean / self.bias).astype(int)
        return self

    def sample(self, *size):
        """Sample random variables from the given distribution

        Parameters
        ----------
        size : int
            The size by axis dimension to sample from the distribution. Each
            size argument passed implies increasing number of dimensions of the
            output.

        Returns
        -------
        numpy.ndarray
            The sampled values from the distribution returned with the given
            size provided.
        """
        return self._state.binomial(self.n_trials, self.bias, size=size)

    def pmf(self, *X):
        """Probability Mass Function

        The probability mass function for the Binomial distribution is given
        by the following function

        .. math::
            {n \choose X} p^X q^{n - X}

        where `n` is the number of independent trials (:code:`n_trials`), `p`
        the bias (:code:`bias`) in favor of a positive event, and `X` the number
        of observed successes.

        The choose notation can be expanded as shown below:

        .. math::
            {n \choose X} = \frac{n!}{X!(n - X)!}

        Note that for numerical accuracy we use the **gamma** function for all
        factorial representations:

        .. math::
            \Gamma(n) = (n - 1)!

        Parameters
        ----------
        X : numpy.ndarray
            1D dataset which falls within the domain of the given distribution
            support. The Binomial distribution expects positive integers only.
            This value is often denoted `k` in the literature.

        Returns
        -------
        numpy.ndarray
            The output probability mass reported elementwise with respect to the
            input data.
        """
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)
        return np.exp(self.log_pmf(X))

    def log_pmf(self, *X):
        """Log Probability Mass Function

        The probability mass function for the Binomial distribution is given
        by the following function

        .. math::
            {n \choose X} p^X q^{n - X}

        where `n` is the number of independent trials (:code:`n_trials`), `p`
        the bias (:code:`bias`) in favor of a positive event, and `X` the number
        of observed successes.

        The choose notation can be expanded as shown below:

        .. math::
            {n \choose X} = \frac{n!}{X!(n - X)!}

        Note that for numerical accuracy we use the **gamma** function for all
        factorial representations:

        .. math::
            \Gamma(n) = (n - 1)!

        Parameters
        ----------
        X : numpy.ndarray
            1D dataset which falls within the domain of the given distribution
            support. The Binomial distribution expects positive integers only.
            This value is often denoted `k` in the literature.

        Returns
        -------
        numpy.ndarray
            The output log-transformed probability mass reported elementwise 
            with respect to the input data.
        """
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # Floor values of X
        X = np.floor(X)

        # Expand all components of log-pmf
        out = (
            sc.gammaln(self.n_trials + 1)
            - (sc.gammaln(X + 1) + sc.gammaln(self.n_trials - X + 1))
            + sc.xlogy(X, self.bias)
            + sc.xlog1py(self.n_trials - X, -self.bias)
        )
        return out

    def cdf(self, *X):
        """Cumulative Distribution Function

        The cumulative distribution function for the Binomial distribution is 
        given below. 

        .. math::
            (n - X) {n \choose X} \int_0^{1 - p} t^{n - X - 1}(1 - t)^X dt

        where `n` is the number of independent trials (:code:`n_trials`), `p`
        the bias (:code:`bias`) in favor of a positive event, and `X` the number
        of observed successes.

        Parameters
        ----------
        X : numpy.ndarray
            1D dataset which falls within the domain of the given distribution
            support. The Binomial distribution expects positive integers only.
            This value is often denoted `k` in the literature.

        Returns
        -------
        numpy.ndarray
            The output cumulative distribution reported elementwise with respect
            to the input data.
        """
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # floor X values
        X = np.floor(X)

        return sc.bdtr(X, self.n_trials, self.bias)

    def log_cdf(self, *X):
        """Log Cumulative Distribution Function

        The cumulative distribution function for the Binomial distribution is 
        given below. 

        .. math::
            (n - X) {n \choose X} \int_0^{1 - p} t^{n - X - 1}(1 - t)^X dt

        where `n` is the number of independent trials (:code:`n_trials`), `p`
        the bias (:code:`bias`) in favor of a positive event, and `X` the number
        of observed successes.

        Parameters
        ----------
        X : numpy.ndarray
            1D dataset which falls within the domain of the given distribution
            support. The Binomial distribution expects positive integers only.
            This value is often denoted `k` in the literature.

        Returns
        -------
        numpy.ndarray
            The output log-transformed cumulative distribution reported 
            elementwise with respect to the input data.
        """
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        return np.log(self.cdf(X))

    def quantile(self, *q):
        """Quantile Function

        Also known as the inverse cumulative distribution function, this 
        function takes known quantiles and returns the associated `X` value from 
        the support domain.

        The quantiles function is given by the incomplete beta function

        .. math::
            I_q(a, b) = \frac{1}{B(a, b)} \int_0^q t^{a - 1}(1 - t)^{b - 1} dt

        This method uses the scipy's bdtrik (a wrapper for the FORTRAN 
        subroutine cdfbin) to reduce the binomial distribution to the cumulative 
        incomplete beta function. The equation is given below

        .. math::
            \sum_{s=a}^{n} p^a (1 - p)^{n - a} = I_p(a, n-a+1)

        Parameters
        ----------
        q : numpy.ndarray, float
            The probabilities within domain [0, 1]

        Returns
        -------
        numpy.ndarray
            The `X` values from the support domain associated with the input
            quantiles.
        """
        # check array for numpy structure
        q = check_array(q, reduce_args=True, ensure_1d=True)

        # get the upper value of X (ceiling)
        X_up = np.ceil(sc.bdtrik(q, self.n_trials, self.bias))

        # get the lower value of X (floor)
        X_down = np.maximum(X_up - 1, 0)

        # recompute quantiles to validate transformation
        q_test = sc.bdtr(X_down, self.n_trials, self.bias)

        # when q_test is greater than true, shift output down
        out = np.where(q_test >= q, X_down, X_up).astype(int)

        # return only in-bound values
        return np.where(self.support.contains(out), out, np.nan)

    @property
    def mean(self):
        return self.n_trials * self.bias

    @property
    def median(self):
        return self.quantile(0.5)[0]

    @property
    def mode(self):
        return self.median

    @property
    def variance(self):
        return self.n_trials * self.bias * (1 - self.bias)

    @property
    def skewness(self):
        n, p, q = self.n_trials, self.bias, 1 - self.bias
        return (q - p) / np.sqrt(n * p * q)

    @property
    def kurtosis(self):
        n, p, q = self.n_trials, self.bias, 1 - self.bias
        return (1 - (6 * p * q)) / (n * p * q)

    @property
    def entropy(self):
        n, p, q = self.n_trials, self.bias, 1 - self.bias
        return 0.5 * logn(2 * np.pi * np.exp(1) * n * p * q, 2)

    @property
    def perplexity(self):
        return np.exp(self.entropy)

    @cache_property
    def support(self):
        return Interval(0, self.n_trials, True, True)

    def _reset(self):
        if hasattr(self, "_n_samples"):
            del self._n_samples
        self.n_trials = None
        self.bias = None
        self._mean = None
        self._variance = None

    def __str__(self):
        return f"Binomial(n_trials={self.n_trials}, bias={self.bias})"

    def __repr__(self):
        return self.__str__()
