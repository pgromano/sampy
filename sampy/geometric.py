import numpy as np
import scipy.special as sc
import warnings

from sampy.distributions import Discrete
from sampy.interval import Interval
from sampy.utils import check_array, cache_property
from sampy.math import nanlog, nanlogn


class ShiftedGeometric(Discrete):
    """ Shifted Geometric Distribution

    The shifted geometric distribution models the probability that the first
    success requires k independent trials. This distribution is an extension of
    the Bernouli distribution (Binomial with fixed single trial), where each
    trial in the geometric case is independent.

    The probability mass function is given as

    .. math::
        (1 - p)^{k - 1} p

    where p is the likelihood of success and k is the number of trials. Within
    this method the value p is represented as the models ..code::`bias` towards
    a success event.

    .. note::
        There are two conflicting (albeit similar) definitions for the
        "geometric" distribution. In an effort to clarify the unique use cases
        of these models, ..code::`sampy` implements both forms under different
        names.

        ..code::`sampy.Geometric`:
            The distribution that defines the probability of k number of
            failures until the first success is observed.
        ..code::`sampy.ShiftedGeometric`:
            The distribution that gives the probability that the first success
            requires k independent trials.
    """

    def __init__(self, bias=0.5, seed=None):
        # safety check bias
        if bias < 0 or bias > 1:
            raise ValueError("Bias must be between [0, 1]")

        self.bias = bias
        self.seed = seed
        self._state = self._set_random_state(seed)

    @classmethod
    def from_data(self, X, seed=None):
        dist = ShiftedGeometric(seed=seed)
        return dist.fit(X)

    def fit(self, X):
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        # check_array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True).astype(float)

        # identify values outside of support
        X[self.support.not_contains(X)] = np.nan

        # first fit
        if not hasattr(self, "_n_samples"):
            self._n_samples = 0

        if self._mean is None:
            self._n_samples += X.shape[0] - np.isnan(X).sum()
            self._mean = np.nanmean(X)
        else:
            # previous values
            prev_size = self._n_samples
            prev_mean = self._mean

            # new values
            curr_size = X.shape[0] - np.isnan(X).sum()
            curr_mean = np.nanmean(X)

            # update size
            self._n_samples = prev_size + curr_size

            # update mean
            self._mean = (
                (prev_mean * prev_size) + (curr_mean * curr_size)
            ) / self._n_samples

        self.bias = 1 / self._mean
        return self

    def sample(self, *size):
        return self._state.geometric(self.bias, size=size)

    def pmf(self, *X):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # return probability mass
        return self.bias * np.power(1 - self.bias, X - 1)

    def log_pmf(self, *X):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # return probability mass
        return np.log(self.pmf(X))

    def cdf(self, *X):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # return cdf
        return 1 - np.power(1 - self.bias, X)

    def log_cdf(self, *X):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # return cdf
        return np.log(self.cdf(X))

    def quantile(self, *q):
        # check array for numpy structure
        q = check_array(q, reduce_args=True, ensure_1d=True)

        # compute quantile
        out = np.ceil(np.log(1 - q) / np.log(1 - self.bias))

        # return safely with bounds check
        return np.where(self.support.contains(out), out, np.nan)

    @property
    def mean(self):
        return 1 / self.bias

    @property
    def median(self):
        return self.quantile(0.5)

    @property
    def mode(self):
        return 1

    @property
    def variance(self):
        return (1 - self.bias) / (self.bias * self.bias)

    @property
    def skewness(self):
        return (2 - self.bias) / np.sqrt(1 - self.bias)

    @property
    def kurtosis(self):
        return 6 + (self.bias * self.bias) / (1 - self.bias)

    @property
    def entropy(self):
        p, q = self.bias, 1 - self.bias
        return (-q * nanlogn(q, 2) - p * nanlogn(p, 2)) / p

    @property
    def perplexity(self):
        return np.exp(self.entropy)

    @cache_property
    def support(self):
        return Interval(1, np.inf, True, False)

    def _reset(self):
        if hasattr(self, "_n_samples"):
            del self._n_samples
        self.bias = None
        self._mean = None

    def __str__(self):
        return f"ShiftedGeometric(bias={self.bias})"

    def __repr__(self):
        return self.__str__()


class Geometric(Discrete):
    """ Geometric Distribution

    The geometric distribution models the probability of k number of  failures 
    until the first success is observed. This distribution is an extension of
    the Bernouli distribution (Binomial with fixed single trial), where each
    trial in the geometric case is independent.

    The probability mass function is given as

    .. math::
        (1 - p)^k p

    where p is the likelihood of success and k is the number of trials. Within
    this method the value p is represented as the models ..code::`bias` towards
    a success event.

    .. note::
        There are two conflicting (albeit similar) definitions for the
        "geometric" distribution. In an effort to clarify the unique use cases
        of these models, ..code::`sampy` implements both forms under different
        names.

        ..code::`sampy.Geometric`:
            The distribution that defines the probability of k number of
            failures until the first success is observed.
        ..code::`sampy.ShiftedGeometric`:
            The distribution that gives the probability that the first success
            requires k independent trials.
    """

    def __init__(self, bias=0.5, seed=None):
        # safety check bias
        if bias < 0 or bias > 1:
            raise ValueError("Bias must be between [0, 1]")

        self.bias = bias
        self.seed = seed
        self._state = self._set_random_state(seed)

    @classmethod
    def from_data(self, X, seed=None):
        dist = Geometric(seed=seed)
        return dist.fit(X)

    def fit(self, X):
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        # check_array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True).astype(float)

        # identify values outside of support
        X[self.support.not_contains(X)] = np.nan

        # first fit
        if not hasattr(self, "_n_samples"):
            self._n_samples = 0

        if self._mean is None:
            self._n_samples += X.shape[0] - np.isnan(X).sum()
            self._mean = np.nanmean(X)
        else:
            # previous values
            prev_size = self._n_samples
            prev_mean = self._mean

            # new values
            curr_size = X.shape[0] - np.isnan(X).sum()
            curr_mean = np.nanmean(X)

            # update size
            self._n_samples = prev_size + curr_size

            # update mean
            self._mean = (
                (prev_mean * prev_size) + (curr_mean * curr_size)
            ) / self._n_samples

        self.bias = 1 / (1 + self._mean)
        return self

    def sample(self, *size):
        return self._state.geometric(self.bias, size=size) - 1

    def pmf(self, *X):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # return probability mass
        return self.bias * np.power(1 - self.bias, X)

    def log_pmf(self, *X):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # return probability mass
        return np.log(self.pmf(X))

    def cdf(self, *X):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # return cdf
        return 1 - np.power(1 - self.bias, X + 1)

    def log_cdf(self, *X):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # return cdf
        return np.log(self.cdf(X))

    def quantile(self, *q):
        # check array for numpy structure
        q = check_array(q, reduce_args=True, ensure_1d=True)

        # compute quantile
        out = np.ceil(np.log(1 - q) / np.log(1 - self.bias)) - 1

        # return safely with bounds check
        return np.where(self.support.contains(out), out, np.nan)

    @property
    def mean(self):
        return (1 - self.bias) / self.bias

    @property
    def median(self):
        return self.quantile(0.5)

    @property
    def mode(self):
        return 0

    @property
    def variance(self):
        return (1 - self.bias) / (self.bias * self.bias)

    @property
    def skewness(self):
        return (2 - self.bias) / np.sqrt(1 - self.bias)

    @property
    def kurtosis(self):
        return 6 + (self.bias * self.bias) / (1 - self.bias)

    @property
    def entropy(self):
        p, q = self.bias, 1 - self.bias
        return (-q * nanlogn(q, 2) - p * nanlogn(p, 2)) / p

    @property
    def perplexity(self):
        return np.exp(self.entropy)

    @cache_property
    def support(self):
        return Interval(0, np.inf, True, False)

    def _reset(self):
        if hasattr(self, "_n_samples"):
            del self._n_samples
        self.bias = None
        self._mean = None

    def __str__(self):
        return f"Geometric(bias={self.bias})"

    def __repr__(self):
        return self.__str__()
