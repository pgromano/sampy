import numpy as np


def check_array(X, squeeze=False):
	try:
		return np.asscalar(X)
	except:
		if squeeze:
			return np.squeeze(X)
		return np.asarray(X)
