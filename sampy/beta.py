import warnings
import numpy as np
import scipy.special as sc
import scipy.optimize as opt

from sampy.distributions import Continuous
from sampy.interval import Interval
from sampy.utils import check_array, cache_property, get_param_permutations, reduce_shape


__all__ = ['Beta']


def _log_loss(params, mu, nu, n):
    # Expand beta parameters
    a, b = params

    # Functions for root solutions
    func1 = n * mu - n * (-sc.psi(a + b) + sc.psi(a))
    func2 = n * nu - n * (-sc.psi(a + b) + sc.psi(b))
    return [func1, func2]


class Beta(Continuous):

    __slots__ = {
        'alpha': 'alpha parameter',
        'beta': 'beta parameter',
        'fit_raise': 'whether or not to raise error on failed fit convergence',
        'seed': 'seed'
    }

    def __init__(self, alpha=1, beta=1, fit_raise="error", seed=None):
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
        if 1 - self.support.contains(X).sum() > 0:
            raise ValueError("Training data not within support domain")

        # first fit
        if not hasattr(self, "_n_samples"):
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
            self._mean = (
                (prev_mean * prev_size) + (curr_mean * curr_size)
            ) / self._n_samples

            # update variance
            self._variance = (
                (prev_variance * prev_size) + (curr_variance * curr_size)
            ) / self._n_samples
            # update mu
            self._mu = ((prev_mu * prev_size) + (curr_mu * curr_size)) / self._n_samples

            # update nu
            self._nu = ((prev_nu * prev_size) + (curr_nu * curr_size)) / self._n_samples

        m, v = self._mean, self._variance
        # m, v = X.mean(), X.var(ddof=0)
        norm = m * (1 - m) / v - 1
        alpha = m * norm
        beta = (1 - m) * norm

        params, info, ier, msg = opt.fsolve(
            _log_loss,
            [alpha, beta],
            args=(self._mu, self._nu, self._n_samples),
            full_output=True,
        )

        if ier != 1:
            if self.fit_raise == "error":
                raise ValueError("Fit failed:\n\n" + msg)
            elif self.fit_raise == "warn":
                if self.alpha is None and self.beta is None:
                    warnings.warn(
                        f"Unable to optimize beta log-loss – parameters will be from method of moments optimization: alpha={alpha}, beta={beta}.\n\nMLE optimization output:\n\n"
                        + msg
                    )
                else:
                    warnings.warn(
                        f"Unable to optimize beta log-loss – parameters will not update: alpha={alpha}, beta={beta}.\n\nMLE optimization output:\n\n"
                        + msg
                    )
            self.alpha, self.beta = alpha, beta
        else:
            self.alpha, self.beta = params

        return self

    def sample(self, *size):
        return self._state.beta(self.alpha, self.beta, size=size)

    def pdf(self, *X, alpha=None, beta=None, keepdims=False):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # get alpha
        if alpha is None:
            alpha = self.alpha

        # get bias
        if beta is None:
            beta = self.beta

        # create grid for multi-paramter search
        X, alpha, beta, shape = get_param_permutations(X, alpha, beta, return_shape=True)

        norm = sc.beta(alpha, beta)
        p = np.power(X, alpha - 1) * np.power(1 - X, beta - 1)
        return reduce_shape(p / norm, shape, keepdims)

    def log_pdf(self, *X, alpha=None, beta=None, keepdims=False):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # get alpha
        if alpha is None:
            alpha = self.alpha

        # get bias
        if beta is None:
            beta = self.beta

        # create grid for multi-paramter search
        X, alpha, beta, shape = get_param_permutations(X, alpha, beta, return_shape=True)

        norm = sc.betaln(alpha, beta)
        p = (alpha - 1) * np.log(X) + (beta - 1) * np.log(1 - X)
        return reduce_shape(p - norm, shape, keepdims)

    def cdf(self, *X, alpha=None, beta=None, keepdims=False):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        # get alpha
        if alpha is None:
            alpha = self.alpha

        # get bias
        if beta is None:
            beta = self.beta

        # create grid for multi-paramter search
        X, alpha, beta, shape = get_param_permutations(X, alpha, beta, return_shape=True)

        out = sc.btdtr(alpha, beta, X)
        return reduce_shape(out, shape, keepdims)

    def log_cdf(self, *X, alpha=None, beta=None, keepdims=False):
        # check array for numpy structure
        X = check_array(X, reduce_args=True, ensure_1d=True)

        return np.log(self.cdf(X, alpha=alpha, beta=beta, keepdims=keepdims))

    def quantile(self, *q, alpha=None, beta=None, keepdims=False):
        # check array for numpy structure
        q = check_array(q, reduce_args=True, ensure_1d=True)

        # get alpha
        if alpha is None:
            alpha = self.alpha

        # get bias
        if beta is None:
            beta = self.beta

        # create grid for multi-paramter search
        q, alpha, beta, shape = get_param_permutations(q, alpha, beta, return_shape=True)

        out = sc.btdtri(alpha, beta, q)
        return reduce_shape(out, shape, keepdims)

    def mean(self, alpha=None, beta=None, keepdims=False):

        # get alpha
        if alpha is None:
            alpha = self.alpha

        # get bias
        if beta is None:
            beta = self.beta

        # create grid for multi-paramter search
        alpha, beta, shape = get_param_permutations(alpha, beta, return_shape=True)

        return reduce_shape(alpha / (alpha + beta), shape, keepdims)

    def median(self, alpha=None, beta=None, keepdims=False):
        return self.quantile(0.5, alpha=alpha, beta=beta, keepdims=keepdims)

    def mode(self, alpha=None, beta=None, keepdims=False):

        # get alpha
        if alpha is None:
            alpha = self.alpha

        # get bias
        if beta is None:
            beta = self.beta

        # create grid for multi-paramter search
        alpha, beta, shape = get_param_permutations(alpha, beta, return_shape=True)

        # alpha > 1 and beta > 1
        out = (alpha - 1) / (alpha - beta - 2)

        # alpha < 1 and beta < 1
        out = np.where(np.logical_and(alpha < 1, beta < 1), np.nan, out)

        # alpha <= 1 and beta > 1
        out = np.where(np.logical_and(alpha <= 1, beta > 1), 0, out)

        # alpha > 1 and beta <= 1
        out = np.where(np.logical_and(alpha > 1, beta <= 1), 1, out)

        return reduce_shape(out, shape, keepdims)

    def variance(self, alpha=None, beta=None, keepdims=False):

        # get alpha
        if alpha is None:
            alpha = self.alpha

        # get bias
        if beta is None:
            beta = self.beta

        # create grid for multi-paramter search
        alpha, beta, shape = get_param_permutations(alpha, beta, return_shape=True)
        a, b = alpha, beta

        # output variance
        out = (a * b) / ((a + b) * (a + b) * (a + b + 1))
        return reduce_shape(out, shape, keepdims)

    def skewness(self, alpha=None, beta=None, keepdims=False):

        # get alpha
        if alpha is None:
            alpha = self.alpha

        # get bias
        if beta is None:
            beta = self.beta

        # create grid for multi-paramter search
        alpha, beta, shape = get_param_permutations(alpha, beta, return_shape=True)
        a, b = alpha, beta

        norm = (a + b + 2) * np.sqrt(a * b)
        out = 2 * (b - a) * np.sqrt(a + b + 1)
        return reduce_shape(out / norm, shape, keepdims)

    def kurtosis(self, alpha=None, beta=None, keepdims=False):

        # get alpha
        if alpha is None:
            alpha = self.alpha

        # get bias
        if beta is None:
            beta = self.beta

        # create grid for multi-paramter search
        alpha, beta, shape = get_param_permutations(alpha, beta, return_shape=True)
        a, b = alpha, beta

        norm = a * b * (a + b + 2) * (a + b + 3)
        out = 6 * ((a + b) * (a + b) * (a + b + 1) - a * b * (a + b + 2))
        return reduce_shape(out / norm, shape, keepdims)

    def entropy(self, alpha=None, beta=None, keepdims=False):

        # get alpha
        if alpha is None:
            alpha = self.alpha

        # get bias
        if beta is None:
            beta = self.beta

        # create grid for multi-paramter search
        alpha, beta, shape = get_param_permutations(alpha, beta, return_shape=True)
        a, b = alpha, beta

        out = sc.betaln(a, b) - (a - 1) * sc.digamma(a)
        out -= (b - 1) * sc.digamma(b)
        out += (a + b - 2) * sc.digamma(a + b)
        return reduce_shape(out, shape, keepdims)

    def perplexity(self, alpha=None, beta=None, keepdims=False):
        return np.exp(self.entropy(alpha, beta, keepdims))

    @cache_property
    def support(self):
        return Interval(0, 1, True, True)

    def _reset(self):
        if hasattr(self, "_n_samples"):
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

