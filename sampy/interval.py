import numpy as np


class Interval:
	""" Interval

	Defines a range from low to high with inclusive or exclusive bounds. Useful
	for evaluating the support of distributions.

	Arguments
	---------
	low: numeric
		The lower bound of the interval
	high: numeric
		The upper bound of the interval
	left: bool (Default: True)
		Whether or not the lower bounds of the interval should be treated
		inclusively. If `low = 0` and `left = True` then 0 would be within the
		interval. If however `left = False` then 0 would fall outside the bound
		but an infinitely small (within computer precision) value greter would 
		be within bounds.
	right: bool (Default: True)
		Whether or not the upper bounds of the interval should be treated
		inclusively. Similar caveats as `left`.
	"""

	def __init__(self, low, high, left=True, right=True):
		if low > high:
			raise ValueError("Interval low must be lesser than high")
		if low == high:
			raise ValueError("Interval low and high cannot be the same value")

		self.low = low
		self.high = high
		self.left = left
		self.right = right

	def contains(self, X):
		""" Vectorized Check of Domain Containment

		This method is similar to use of the `in` operator (e.g. 
		`5 in Interval()`) however is implemented to vectorize across array-like
		objects. 

		Parameters
		----------
		X : numeric or array-like
			The individual or container of values to be evaluated. 

		Returns
		-------
		bool or array of bool
			A boolean value of whether or not the values in X are within the
			given support domain. If a numeric value (int or float) then a 
			single bool is returned. If an array-like object is passed, then
			a numpy.ndarray of type bool is returned with each value 
			representing elementwise values.
		"""

		if self.left and self.right:
			return np.logical_and(self.low <= X, X <= self.high)
		elif self.left and not self.right:
			return np.logical_and(self.low <= X, X < self.high)
		elif not self.left and self.right:
			return np.logical_and(self.low < X, X <= self.high)
		elif not self.left and not self.right:
			return np.logical_and(self.low < X, X < self.high)

	def __lt__(self, val):
		if self.left:
			return val.__gt__(self.low)
		return val.__ge__(self.low)

	def __le__(self, val):
		return val.__gt__(self.high) or val in self

	def __gt__(self, val):
		if self.right:
			return val.__lt__(self.high)
		return val.__le__(self.high)

	def __ge__(self, val):
		return val.__lt__(self.low) or val in self

	def __contains__(self, val):

		if not isinstance(val, (int, float)):
			raise ValueError("Cannot compare non-numeric data in distribution support")

		if self.left and self.right:
			return self.low <= val <= self.high
		elif self.left and not self.right:
			return self.low <= val < self.high
		elif not self.left and self.right:
			return self.low < val <= self.high
		elif not self.left and not self.right:
			return self.low < val < self.high

	@property
	def __name__(self):
		return self.__class__.__name__

	def __str__(self):
		# define left bracket (lb)
		if self.left:
			lb = '['
		else:
			lb = '('

		# define right bracket (rb)
		if self.right:
			rb = ']'
		else:
			rb = ')'
		return f"{lb}{self.low}, {self.high}{rb}"

	def __repr__(self):
		return self.__str__()
