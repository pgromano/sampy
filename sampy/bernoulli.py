import warnings
import numpy as np
import scipy.special as sc

from sampy.distributions import Discrete
from sampy.interval import Interval
from sampy.utils import check_array, cache_property
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
	invalid : str, {'error', 'warn', 'ignore'}
		How the fit method should handle out of domain errors or other invalid
		values. On 'error' the method will return ValueError and end. On 'warn'
		a warning will notify the user of the issue and actions taken to 
		mitigate. On 'ignore' the user is neither informed nor does the code
		error out, proceed with caution.
	"""

    def __init__(self, bias=0.5, invalid="error", seed=None):
        self.bias = bias
        self.seed = seed
        if invalid not in {"error", "warn", "ignore"}:
            raise ValueError(f"Unable to interpret `invalid = {invalid}`")
        self.invalid = invalid
        self._state = self._set_random_state(seed)

    @classmethod
    def from_data(self, X, invalid="error", seed=None):
        """Initialize distribution from data

        Parameters
        ----------
        X : numpy.ndarray
            1D dataset which falls within the domain of the given distribution
            support. The Bernoulli distribution expects series of 0 or 1 only.
        invalid : str, {'error', 'warn', 'ignore'}
            How the fit method should handle out of domain errors or other invalid
            values. On 'error' the method will return ValueError and end. On 'warn'
            a warning will notify the user of the issue and actions taken to 
            mitigate. On 'ignore' the user is neither informed nor does the code
            error out, proceed with caution.
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

        dist = Bernoulli(invalid=invalid, seed=seed)
        return dist.fit(X)

    def fit(self, X):
        """Fit model to data

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
        if X.dtype != int:
            if self.invalid == "error":
                raise ValueError(
                    "Bernoulli distribution must be dtype intbut found {X.dtype}"
                )
            elif self.invalid == "warn":
                warnings.warn(
                    f"Bernoulli distribution should be dtype int but found {X.dtype}. Attempting to cast to int",
                    category=UserWarning,
                )
        X = X.astype(float)

        # identify values outside of support
        invalid = (1 - self.support.contains(X)).astype(bool)
        if np.sum(invalid) > 0:
            if self.invalid == "error":
                raise ValueError("Bernoulli must be 0 or 1 values only")
            elif self.invalid == "warn":
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

    def pmf(self, *X):
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

        Returns
        -------
        numpy.ndarray
            The output probability mass reported elementwise with respect to the
            input data.
        """

        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # probability mass
        out = np.where(X == 1, self.bias, 1 - self.bias)

        # check bounds
        out = np.where(self.support.contains(out), out, np.nan)
        return out

    def log_pmf(self, *X):
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

        return np.log(self.pmf(X))

    def cdf(self, *X):
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

        # when X in bounds: 1 - p, else 0
        out = np.where(self.support.contains(X), 1 - self.bias, 0)

        # when X >= 1 then 1
        out = np.where(X >= 1, 1, out)
        return out

    def log_cdf(self, X):
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

        return np.log(self.cdf(X))

    def quantile(self, *q):
        """Quantile Function

        Also known as the inverse cumulative Distribution function, this function
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

        out = np.ceil(sc.bdtrik(q, 1, self.bias))
        return np.where(self.bias >= q, 0, out)

    @property
    def mean(self):
        return self.bias

    @property
    def median(self):
        return self.quantile(0.5)[0]

    @property
    def mode(self):
        if self.bias == 0.5:
            return np.nan
        return self.median

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

    @property
    def entropy(self):
        p, q = self.bias, 1 - self.bias
        return -q * np.log(q) - p * np.log(p)

    @property
    def perplexity(self):
        return np.exp(self.entropy)

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
