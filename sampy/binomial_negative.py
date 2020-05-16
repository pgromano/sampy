import numpy as np
import scipy.special as sc
import warnings

from sampy.distributions import Discrete
from sampy.interval import Interval
from sampy.utils import check_array, cache_property
from sampy.math import logn, _handle_zeros_in_scale, nanlog


class NegativeBinomial(Discrete):
    def __init__(self, n_success=1, bias=0.5, seed=None):

        # safety check n_success
        if isinstance(n_success, float):
            warnings.warn("Number of success should be an integer. Casting to int")

        if np.round(n_success).astype(int) != n_success:
            raise ValueError("Number of success must be an integer value")
        else:
            n_success = np.round(n_success).astype(int)

        if n_success < 0:
            raise ValueError("Number of successes must be greater than or equal to 0")

        # safety check bias
        if bias < 0 or bias > 1:
            raise ValueError("Bias must be between [0, 1]")

        self.n_success = n_success
        self.bias = bias
        self.seed = seed
        self._state = self._set_random_state(seed)

    def sample(self, *size):
        return self._state.negative_binomial(self.n_success, self.bias, size=size)

    def pmf(self, *X):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)
        return np.exp(self.log_pmf(X))

    def log_pmf(self, *X):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # floor values of X
        X = np.floor(X)

        # alias n_success
        k = self.n_success

        # expand all components of log-pmf
        (k + X - 1, k)
        out = (
            sc.gammaln(k + X)
            - (sc.gammaln(k + 1) + sc.gammaln(X))
            + X * nanlog(1 - self.bias)
            + k * nanlog(self.bias)
        )
        return out

    def cdf(self, *X):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # floor X values
        X = np.floor(X)

        return sc.betainc(self.n_success, X + 1, self.bias)

    def log_cdf(self, *X):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        return np.log(self.cdf(X))

    def quantile(self, *q):
        # check array for numpy structure
        q = check_array(q, reduce_args=True, ensure_1d=True)

        # get the upper value of X (ceiling)
        X_up = np.ceil(sc.nbdtrik(q, self.n_success, self.bias))

        # get the lower value of X (floor)
        X_down = np.maximum(X_up - 1, 0)

        # recompute quantiles to validate transformation
        q_test = sc.nbdtr(X_down, self.n_success, self.bias)

        # when q_test is greater than true, shift output down
        out = np.where(q_test >= q, X_down, X_up).astype(int)

        # return only in-bound values
        return np.where(self.support.contains(out), out, np.nan)

    @property
    def n_fail(self):
        return self.n_success * ((1 / self.bias) - 1)

    @property
    def mean(self):
        return (self.bias * self.n_fail) / (1 - self.bias)

    @property
    def median(self):
        return self.quantile(0.5)[0]

    @property
    def mode(self):
        if self.n_fail <= 1:
            return 0
        return np.floor((self.bias * (self.n_fail - 1)) / (1 - self.bias))

    @property
    def variance(self):
        return self.bias * self.n_fail / ((1 - self.bias) * (1 - self.bias))

    @property
    def skewness(self):
        return (1 + self.bias) / np.sqrt(self.bias * self.n_fail)

    @property
    def kurtosis(self):
        return 6 / self.n_fail + (
            (1 - self.bias) * (1 - self.bias) / (self.bias * self.n_fail)
        )

    @cache_property
    def support(self):
        return Interval(0, np.inf, True, False)

    def __str__(self):
        return f"NegativeBinomial(n_success={self.n_success}, bias={self.bias})"

    def __repr__(self):
        return self.__str__()
