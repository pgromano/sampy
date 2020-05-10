import numpy as np
import scipy.special as sc

from sampy.gamma import Gamma
from sampy.utils import check_array


class ChiSquared(Gamma):

	def __init__(self, dof=1, seed=None):
		self.dof = dof
		super(ChiSquared, self).__init__(0.5 * dof, 0.5, seed=seed)

	def partial_fit(self, X):

		# check array for numpy structure
		X = check_array(X, squeeze=True)

		super(ChiSquared, self).partial_fit(X)
		self.dof = np.round(2 * self.shape).astype(int)
		return self

	@property
	def mean(self):
		return float(self.dof)

	@property
	def median(self):
		return self.quantile(0.5)

	@property
	def mode(self):
		return max([self.dof - 2.0, 0.0])

	@property
	def variance(self):
		return 2.0 * self.dof

	@property
	def skewness(self):
		return np.sqrt(8 / self.dof)

	@property
	def kurtosis(self):
		return 12.0 / self.dof

	@property
	def entropy(self):
		# reduced degrees of freed
		k = self.dof / 2.0
		return k + np.log(2 * sc.gamma(k)) + (1 - k) * sc.psi(k)

	@property
	def perplexity(self):
		return np.exp(self.entropy)

	@property
	def support(self):
		if self.dof == 1:
			return Interval(0, np.inf, False, False)
		return Interval(0, np.inf, True, False)

	def __str__(self):
		return f"Chi2(dof={self.dof})"

	def __repr__(self):
		return self.__str__()
