# `sampy`

`sampy` is a python package built on `numpy` for efficient computation of statistical properties as well as fast sampling. While several packages already exist for statistics in Python, `sampy` aims to be a simple API  for distributions built on limited dependencies.

# Generic API

Every distribution method in `sampy` initializes with the only the most immediate parameters for the distribution and a seed. Take the example of the normal distribution below.

```python
dist = sampy.Normal(center=0, scale=1, seed=42)
```

One feature we've added is the ability to safely hash strings, so that you can seed your distributions with helpful comments.

```python
dist = sampy.Normal(0, 1, seed='Docs example of sampy seeds')
dist.seed
>>> 'Docs example of sampy seeds'
```

One gap that generally persists across is the ability to conveniently pass data to _fit_ a distribution. `sampy` aims to provide not only the ability to fit a distribution _once_, but to continue to refine the parameters given new data. 

```python
# sample from true distribution
X = dist.sample(10)

# train model
model = sampy.Normal().fit(X)
print(model)
>>> Normal(center=0.302050479042494, scale=0.9676584235555014)

# continue fitting data
for x in dist.sample(100, 10):
    model.partial_fit(x)
print(model)
>>> Normal(center=0.0022595260156181842, scale=0.9572802903417377)
```

To simplify the API further, we have also added the class method `from_data` to quickly intialize a new model distribution without the extra call initialization.

```python
model = sampy.Normal.from_data(X, seed='Example of from_data class method')
```

**Note**: Both `from_data` and `fit` are identical in that they clear the distribution of any pre-defined parameters. )