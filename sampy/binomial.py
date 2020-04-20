import numpy as np
import scipy.special as sc

from .distributions import Discrete
from .utils import _handle_zeros_in_scale, check_array


class Binomial(Discrete):
	def __init__(self, rate=1, seed=None):
		self.rate = rate
		self.seed = seed
		self._state = self._set_random_state(seed)
