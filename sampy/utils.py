import numpy as np


__all__ = [
	'check_array',
	'set_random_state',
	'cache_property',
]


def check_array(X, ensure_1d=False, ensure_2d=False, squeeze=False, 
				atleast_2d=False, feature_axis='col', reduce_args=False, 
				dtype=None):
	""" Check Array

	Parameters
	----------
	X : array-like or numeric
		The input data to standardize as a numpy array
	squeeze : bool, optional
		Whether or not the array should be "squeezed". This results in 
		flattening all size 1 dimensions, by default False.

	Returns
	-------
	numpy.ndarray
		The cleaned data as a numpy.ndarray
	"""

	if ensure_1d and ensure_2d:
		raise ValueError("Cannot ensure 1D and 2D array")

	if ensure_1d and atleast_2d:
		raise ValueError("Ambiguous expectation: ensure_1d and atleast_2d")
	
	if squeeze and atleast_2d:
		raise ValueError("Ambiguous expectation: squeeze and atleast_2d")

	if reduce_args:
		if len(X) == 1:
			X = X[0]

		if not hasattr(X, '__iter__') or isinstance(X, str):
			X = np.array([X])
		elif hasattr(X, '__iter__') and not isinstance(X, str):
			X = np.array(X)

	if hasattr(X, '__iter__') and not isinstance(X, str) and not isinstance(X, np.ndarray):
		X = np.array(X)

	if squeeze:
		X = np.squeeze(X)

	if atleast_2d and np.ndim(X) == 1:
		if feature_axis == 'row' or feature_axis == 0:
			X = np.atleast_2d(X)
		elif feature_axis == 'col' or feature_axis == 1:
			X = np.atleast_2d(X).T
		else:
			raise ValueError(
				f"Unable to intepret `feature_axis = '{feature_axis}'`")

	if ensure_1d and np.ndim(X) != 1:
		raise ValueError("Array must be 1D")

	if ensure_2d and np.ndim(X) != 2:
		raise ValueError("Array must be 2D")

	return X


def set_random_state(seed=None):
	if isinstance(seed, np.random.RandomState):
		return seed

	if isinstance(seed, str):
		seed = hash(seed) & ((1 << 32) - 1)

	return np.random.RandomState(seed)



class cache_property:
	def __init__(self, method):
		self.method = method
		self.cache_name = "_{}".format(method.__name__)

	def __get__(self, instance, *args, **kwargs):
		
		if hasattr(instance, self.cache_name):
			return getattr(instance, self.cache_name)
		setattr(instance, self.cache_name, self.method(instance))
		return getattr(instance, self.cache_name)

