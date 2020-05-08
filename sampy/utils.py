import numpy as np


__all__ = [
	'check_array',
	'Interval'
]


def check_array(X, squeeze=False, dtype=None):
	"""[summary]

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
	try:
		X = np.asscalar(X)
		if dtype is None:
			return X
		return X.astype(dtype)
	except:
		if squeeze:
			X = np.squeeze(X)
		else:
			X = np.asarray(X)
		
		if dtype is None:
			return X
		return X.astype(dtype)


def set_random_state(self, seed=None):
	if isinstance(seed, np.random.RandomState):
		return seed

	if isinstance(seed, str):
		seed = hash(seed) & ((1 << 32) - 1)

	return np.random.RandomState(seed)
