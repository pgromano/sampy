import numpy as np


__all__ = [
	'check_array',
	'Interval'
]


def check_array(X, squeeze=False):
	try:
		return np.asscalar(X)
	except:
		if squeeze:
			return np.squeeze(X)
		return np.asarray(X)


class Interval:
	def __init__(self, low, high, left_inclusive=True, right_inclusive=True):
		if low > high:
			raise ValueError("Interval low must be lesser than high")
		if low == high:
			raise ValueError("Interval low and high cannot be the same value")

		self.low = low
		self.high = high
		self.left_inclusive = left_inclusive
		self.right_inclusive = right_inclusive

	def __lt__(self, val):
		if self.left_inclusive:
			return val.__gt__(self.low)
		return val.__ge__(self.low)

	def __le__(self, val):
		return val.__gt__(self.high) or val in self

	def __gt__(self, val):
		if self.right_inclusive:
			return val.__lt__(self.high)
		return val.__le__(self.high)

	def __ge__(self, val):
		return val.__lt__(self.low) or val in self

	def __contains__(self, val):
		if self.left_inclusive and self.right_inclusive:
			return val >= self.low and val <= self.high
		elif self.left_inclusive and not self.right_inclusive:
			return val >= self.low and val < self.high
		elif not self.left_inclusive and self.right_inclusive:
			return val > self.low and val <= self.high
		elif not self.left_inclusive and not self.right_inclusive:
			return val > self.low and val < self.high

	@property
	def __name__(self):
		return self.__class__.__name__

	def __str__(self):
		# define left bracket (lb)
		if self.left_inclusive:
			lb = '['
		else:
			lb = '('

		# define right bracket (rb)
		if self.right_inclusive:
			rb = ']'
		else:
			rb = ')'
		return f"{lb}{self.low}, {self.high}{rb}"

	def __repr__(self):
		return self.__str__()
