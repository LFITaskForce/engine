# engine
[![Gitter](https://badges.gitter.im/LFITaskForce/Engine.svg)](https://gitter.im/LFITaskForce/Engine?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

A Generic Framework for Likelihood-Free Inference.

This proposal relies heavily on Edward2 to define and manipulate the models.
Here is a short example which can also be found in `run_gaussian_noise.py`
```python
def model(num_samples):
    # Define priors
    mu = ed.Normal(loc=0., scale=1., name='mu', sample_shape=(num_samples, 1))
    # Simulator
    obs = ed.Normal(loc=mu, scale=0.1, name='obs')
    # Return the observations
    return obs

# Create mock observation
obs = np.array([[1.]])

# Paramaters over which to run the inference
params = ['mu']

# Instantiate and run inference engine
engine = RejectionABC(model, params, obs=obs)
posterior, samples = engine.run(n_samples=1000000)
```

## Requirements

This version of the engine relies on TensorFlow 2.0 and TensorFlow Probability,
which requires using the nightly builds. Using a virtualenv is a good idea ;-)

```bash
$ virtualenv --system-site-packages -p python3 ~/.venv/tf-nightly
$ source ~/.venv/tf-nightly/bin/activate
$ pip install --upgrade tf-nightly-2.0-preview tfp-nightly
```

A useful tip when working with Jupyter notebook is the possibility to define a
kernel running in the virtualenv:
```bash
$ python -m ipykernel install --user --name=tf-nightly
```
which will install a new Jupyter kernel in `~/.local/share/jupyter/kernels/tf-nightly`.
Just edit the `kernel.json` file there so that it looks like the following:
```json
{
 "argv": [
  "/home/francois/.venv/tf-nightly/bin/python3",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "tf-nightly",
 "language": "python"
}
```
