import numpy as np


def _handle_zeros_in_scale(scale, copy=True):
	''' Makes sure that whenever scale is zero, we handle it correctly.
	This happens in most scalers when we have constant features.

	This is copied from scikit-learn under LICENSE

	see: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/preprocessing/_data.py
	'''

	# if we are fitting on 1D arrays, scale might be a scalar
	if np.isscalar(scale):
		if scale == .0:
			scale = 1.
		return scale
	elif isinstance(scale, np.ndarray):
		if copy:
			# New array to avoid side-effects
			scale = scale.copy()
		scale[scale == 0.0] = 1.0
		return


def logn(X, base):
	return np.log(X) / np.log(base)


def nanlog(X):
	return np.where(X == 0, 0, np.log(X))


def nanlogn(X, base):
	return np.where(X == 0, 0, logn(X, base))
