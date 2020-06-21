import warnings
import numpy as np
import scipy.special as sc

from sampy.distributions import Discrete
from sampy.interval import Interval
from sampy.utils import check_array, cache_property, get_param_permutations, reduce_shape
from sampy.math import _handle_zeros_in_scale


__all__ = ['Bernoulli']


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

    __slots__ = {
        'bias': 'Bias in favor of positive event', 
        'seed': 'Random number generator seed'
    }

    def __init__(self, bias=0.5,seed=None):
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
            support. The Bernoulli distribution expects series of 0 or 1 only.
        seed : int, str or None, optional
            The seed by which to set the random number generator for the 
            `sample` method, by default None. A seed set to None will use the
            default clock time (see numpy.random.RandomState for more details)
            and an integer will set the seed directly. Strings will be hashed to 
            an integer representation which can be a helpful description of the 
            distribution use or associated experiment. 

        Returns
        -------
        sampy.Bernoulli
            The fitted Bernoulli distribution model 
        """

        dist = Bernoulli(seed=seed)
        return dist.fit(X)

    def fit(self, X):
        """Fit model to data

        Parameters are fit using a method of moments estimation. The bias
        parameter can be estimated from the empirical mean sampled data.

        Parameters
        ----------
        X : numpy.ndarray
            1D dataset which falls within the domain of the given distribution
            support. The Bernoulli distribution expects series of 0 or 1 only.

        Returns
        -------
        sampy.Bernoulli
            The fitted Bernoulli distribution model
        """

        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """Incremental fit on a batch of samples

        Parameters are updated using a method of moments estimation. The bias
        parameter can be estimated from the empirical mean sampled data.

        Parameters
        ----------
        X : numpy.ndarray
            1D dataset which falls within the domain of the given distribution
            support. The Bernoulli distribution expects series of 0 or 1 only.

        Returns
        -------
        sampy.Bernoulli
            The fitted Bernoulli distribution model
        """

        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)
        if X.dtype.kind != 'i':
            warnings.warn(
                f"Bernoulli distribution should be dtype int but found {X.dtype}. Attempting to cast to int",
                category=UserWarning,
            )
        X = X.astype(float)

        # identify values outside of support
        invalid = (1 - self.support.contains(X)).astype(bool)
        if np.sum(invalid) > 0:
            warnings.warn(
                f"Training data outside support {self.support}. Will train ignoring values outside domain",
                category=UserWarning,
            )
        X[invalid] = np.nan

        # first fit
        if not hasattr(self, "_n_samples"):
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
            self.bias = (
                (prev_rate * prev_size) + (curr_rate * curr_size)
            ) / self._n_samples

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

        return self._state.binomial(1, self.bias, size=size)

    def pmf(self, *X, bias=None, keepdims=False):
        """Probability Mass Function

        The probability mass function for the Bernoulli distribution is given
        by two cases. 

        .. math::
            \begin{cases}
                1-p  &\text{if } X = 0\\
                p    &\text{if } X = 1
            \end{cases}

        where `p` is the :code:`bias` in favor of a positive event

        Parameters
        ----------
        X : numpy.ndarray, int
            1D dataset which falls within the domain of the given distribution
            support. The Bernoulli distribution expects series of 0 or 1 only.
            This value is often denoted `k` in the literature.
        bias : float, optional
            The bias or the likelihood of a positive event, by default 0.5. 
            The values within this method assume that negative class will be 0
            and positive class 1.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.

        Returns
        -------
        numpy.ndarray
            The output probability mass reported elementwise with respect to the
            input data.
        """

        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # get bias
        if bias is None:
            bias = self.bias

        # create grid for multi-paramter search
        X, bias, shape = get_param_permutations(X, bias, return_shape=True)

        # probability mass
        out = np.where(X == 1, bias, 1 - bias)

        # check bounds
        in_bounds = np.logical_and(out >= 0, out <= 1)
        out = np.where(in_bounds, out, np.nan)
        return reduce_shape(out, shape, keepdims)

    def log_pmf(self, *X, bias=None, keepdims=False):
        """Log Probability Mass Function

        The probability mass function for the Bernoulli distribution is given
        by two cases. 

        .. math::
            \begin{cases}
                1-p  &\text{if } X = 0\\
                p    &\text{if } X = 1
            \end{cases}

        where `p` is the :code:`bias` in favor of a positive event

        Parameters
        ----------
        X : numpy.ndarray, int
            1D dataset which falls within the domain of the given distribution
            support. The Bernoulli distribution expects series of 0 or 1 only.
            This value is often denoted `k` in the literature.

        Returns
        -------
        numpy.ndarray
            The output log transformed probability mass reported elementwise 
            with respect to the input data.
        """

        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        return np.log(self.pmf(X, bias=bias, keepdims=keepdims))

    def cdf(self, *X, bias=None, keepdims=False):
        """Cumulative Distribution Function

        The cumulative distribution function for the Bernoulli distribution is 
        given by three cases. 

        .. math::
            \begin{cases}
                0      &\text{if } X \leq 0
                1 - p  &\text{if } 0 \leq X \lt 1\\
                1      &\text{if } X \geq 1
            \end{cases}

        where `p` is the :code:`bias` in favor of a positive event

        Parameters
        ----------
        X : numpy.ndarray, int
            1D dataset which falls within the domain of the given distribution
            support. The Bernoulli distribution expects series of 0 or 1 only.
            This value is often denoted `k` in the literature.

        Returns
        -------
        numpy.ndarray
            The output cumulative distribution reported elementwise with respect to 
            the input data.
        """

        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # get bias
        if bias is None:
            bias = self.bias

        # create grid for multi-paramter search
        X, bias, shape = get_param_permutations(X, bias, return_shape=True)

        # when X in bounds: 1 - p, else 0
        in_bounds = np.logical_and(X >= 0, X <= 1)
        out = np.where(in_bounds, 1 - bias, 0)

        # when X >= 1 then 1
        out = np.where(X >= 1, 1, out)
        return reduce_shape(out, shape, keepdims)

    def log_cdf(self, *X, bias=None, keepdims=False):
        """Log Cumulative Distribution Function

        The cumulative distribution function for the Bernoulli distribution is 
        given by three cases. 

        .. math::
            \begin{cases}
                0      &\text{if } X \leq 0
                1 - p  &\text{if } 0 \leq X \lt 1\\
                1      &\text{if } X \geq 1
            \end{cases}

        where `p` is the :code:`bias` in favor of a positive event

        Parameters
        ----------
        X : numpy.ndarray, int
            1D dataset which falls within the domain of the given distribution
            support. The Bernoulli distribution expects series of 0 or 1 only.
            This value is often denoted `k` in the literature.

        Returns
        -------
        numpy.ndarray
            The output log transformed cumulative distribution reported elementwise
            with respect to the input data.
        """

        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        return np.log(self.cdf(X, bias=bias, keepdims=keepdims))

    def quantile(self, *q, bias=None, keepdims=False):
        """Quantile Function

        Also known as the inverse cumulative distribution function, this function
        takes known quantiles and returns the associated `X` value from the 
        support domain.

        .. math::
            \begin{cases}
                0      &\text{if } 0 \leq q \lt p
                1      &\text{if } p \leq q \lt 1
            \end{cases}

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

        # get bias
        if bias is None:
            bias = self.bias

        # create grid for multi-paramter search
        q, bias, shape = get_param_permutations(q, bias, return_shape=True)

        # get the upper value of X (ceiling)
        X_up = np.ceil(sc.bdtrik(q, 1, bias))

        # get the lower value of X (floor)
        X_down = np.maximum(X_up - 1, 0)

        # recompute quantiles to validate transformation
        q_test = sc.bdtr(X_down, 1, bias)

        # when q_test is greater than true, shift output down
        out = np.where(q_test >= q, X_down, X_up).astype(int)

        # boundary test
        out = np.where(q == 1, 1, out)

        # out of bounds
        out_bounds = np.logical_or(q < 0, q > 1)
        out = np.where(out_bounds, np.nan, out)

        return reduce_shape(out, shape, keepdims)

    def mean(self, bias=None):

        # get bias
        if bias is None:
            bias = self.bias

        return bias

    def median(self, bias=None):
        return self.quantile(0.5, bias=bias, keepdims=False)

    def mode(self, bias=None):
        # get bias
        if bias is None:
            bias = self.bias

        out = self.median(bias, keepdims)
        return np.where(out == 0.5, np.nan, out)

    def variance(self, bias=None):

        # get bias
        if bias is None:
            bias = self.bias
        else:
            bias = check_array(bias, ensure_1d=True)

        return bias * (1 - bias)

    def skewness(self, bias=None):

        # get bias
        if bias is None:
            bias = self.bias
        else:
            bias = check_array(bias, ensure_1d=True)

        p, q = bias, 1 - bias
        return (q - p) / np.sqrt(p * q)

    def kurtosis(self, bias=None):

        # get bias
        if bias is None:
            bias = self.bias
        else:
            bias = check_array(bias, ensure_1d=True)

        p, q = bias, 1 - bias
        return (1 - 6 * p * q) / (p * q)

    def entropy(self, bias=None):

        # get bias
        if bias is None:
            bias = self.bias
        else:
            bias = check_array(bias, ensure_1d=True)

        p, q = bias, 1 - bias
        return -q * np.log(q) - p * np.log(p)

    def perplexity(self, bias=None):

        # get bias
        if bias is None:
            bias = self.bias
        else:
            bias = check_array(bias, ensure_1d=True)

        return np.exp(self.entropy(bias))

    @cache_property
    def support(self):
        return Interval(0, 1, True, True)

    def _reset(self):
        if hasattr(self, "_n_samples"):
            del self._n_samples
        self.bias = None

    def __str__(self):
        return f"Bernoulli(bias={self.bias})"

    def __repr__(self):
        return self.__str__()