# `GaussianNoise()`

Homoscedastic Gaussian noise around mean function in N dimensions. Use the
argument `fun` to set the mean function, some examples are given below.


## Mean functions

```python
cubic = lambda theta: ((1.5*theta+0.5)**3) / 200
sine = lambda theta: np.sin(theta+np.pi/8)
```
