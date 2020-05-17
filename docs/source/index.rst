.. sampy documentation master file, created by
   sphinx-quickstart on Sun May 17 13:41:17 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

sampy
=================================

`sampy` is a python package built to be a simple API for statistical 
distributions. The goal is to provide analysts with a simple tool to model data 
and sample known or empirical distributions.

Installation
=================================

`sampy` is distributed on PyPi and can be installed with pip.

.. code-block:: bash

   pip install sampy

Generic API
=================================

Every distribution method in `sampy` initializes with the only the most 
immediate parameters for the distribution and a seed. Take the example of the 
normal distribution below.

.. code-block:: python
   
   # standard normal distribution
   normal = sampy.Normal(center=0, scale=1)

   # Poisson process
   poisson = sampy.Poisson(rate=1)

   # coin toss distribution
   coin = sampy.Binomial(n_trials=1, bias=0.5)

   # rolling dice distribution
   die = sampy.DiscreteUniform(low=1, high=6, high_inclusive=True)

Setting Seeds
-----------------

`sampy` sets multiple methods for intializing the random number generator state. 
The RNG state is built from the `RandomState` object from `numpy` and as such 
can take integer values to explicitly set the seed. The random state is set from 
within distribution classes, however we'll illustrate this with the underlying 
utility function.

.. code-block:: python

   # seed set by system clock
   sampy.utils.set_random_state(None)

   # seed set by integer
   sampy.utils.set_random_state(42)

One feature added is the ability to safely hash strings, so that seeds can be 
set as helpful comments or descriptions.

.. code-block:: python

   dist = sampy.Binomial(1, 0.5, seed='docs example of sampy seeds')
   dist.seed
   >>> 'docs example of sampy seeds'

Fitting Distribution Models
--------------------------------------------------------

`sampy` provides a `scikit-learn` api style that allows users to `fit` data as 
well as incrementally update parameters via the `partial_fit` method.

.. code-block:: python

   # sample from true distribution
   true = sampy.Normal(0, 1, seed='sampy docs examples for model fit')
   X = true.sample(10)

   # train model
   model = sampy.Normal().fit(X)
   model
   >>> Normal(center=-0.19680447372083085, scale=0.8069214766303398)

This shows that the method of moments implementation decently in limited data, 
however 10 data points is very limited and in real engineering systems data can 
be streamed. A model should be able to update as new data is introduced.

.. code-block:: python

   # continue fitting data
   for x in true.sample(100, 10):
       model.partial_fit(x)
   model
   >>> Normal(center=-0.0588523276136086, scale=0.9130479685560422)

To simplify the API further, we have also added the class method `from_data` to 
quickly intialize a new model distribution without the extra call 
initialization.

.. code-block:: python

   sampy.Normal.from_data(X, seed='Example of from_data class method')

.. note::
   
   Both `from_data` and `fit` are identical in implementation that they 
   clear the distribution of any pre-defined parameters. The `from_data` method 
   can be beneficial in cases when parameters have no prior estimate.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
