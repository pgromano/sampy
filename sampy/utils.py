import numpy as np


__all__ = [
	'check_array',
	'Interval'
]


def check_array(X, squeeze=False):
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
		return np.asscalar(X)
	except:
		if squeeze:
			return np.squeeze(X)
		return np.asarray(X)