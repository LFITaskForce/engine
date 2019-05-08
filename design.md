# Design document for LFI Engine

## Objective

Provide a simple and efficient framework for performing Likelihood-Free Inference on top of a probabilistic programming language.  


## Related work  

Related projects can be broken down in two classes, generic inference frameworks and LFI implementations.


### Inference frameworks

- [PyMC3](https://docs.pymc.io/): Probabilistic Programming in Python: Bayesian Modeling and Probabilistic Machine Learning with Theano.
- [PyMC4](https://github.com/pymc-devs/pymc4): A high-level probabilistic programming interface for TensorFlow Probability.
- [Pyro](http://pyro.ai/): Pyro is a universal probabilistic programming language (PPL) written in Python and supported by PyTorch on the backend.
- [TensorFlow Probability](https://github.com/tensorflow/probability): Probabilistic reasoning and statistical analysis in TensorFlow.

### LFI engines

- [AstroABC](https://github.com/EliseJ/astroABC)
- [Carl](https://github.com/diana-hep/carl)
- [CosmoABC](https://github.com/COINtoolbox/CosmoABC)
- [Delfi](https://github.com/mackelab/delfi): SNPE
- [ELFI](https://github.com/elfi-dev/elfi): ABC, ABC-SMC, BOLFI
- [Hypothesis](https://github.com/montefiore-ai/hypothesis): ABC, AVO, (LF)-MCMC
- [PyDelfi](https://github.com/justinalsing/pydelfi): SNL


## Design overview

Framework should remain lightweight, sticking as much as possible to existing conventions and tools from most closely related back-end PPL.

We sketch out how using the LFI engine may look like with different inference frameworks as backends.


### Pyro

Central to designing inference algorithms with Pyro are effect handlers applied to stochastic functions. Models are written as stochastic functions that contain primitives causing side effects, see [Pyro's examples and tutorials](http://pyro.ai/examples/) for an introduction. In order to understand how this works internally, see [Poutine: A Guide to Programming with Effect Handlers in Pyro](http://pyro.ai/examples/effect_handlers.html) and [minipyro.py](https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/minipyro.py).

Step-by-step, we'll develop a proposal for a model design adhering to Pyro's principles --- while trying to provide an intuitive interface for the user.


#### External simulators

For starters, say, we got code for an external simulator: an arbitrary  external program implementing the data-generating process we want to do inference for. We have a Python interface to this code that inputs and outputs NumPy arrays.

```python
def model():
	inputs = torch.tensor([[...], [...]])  # (batch, dim_inputs)
	outputs = torch.tensor([my_simulator(inputs.numpy())])  # (batch, ...)
```

This can be made look nicer through a decorator handling type conversion:

```python
def model():
	inputs = torch.tensor([[...], [...]])
	outputs = numpy_simulator(my_simulator)(inputs)
```

Inside `model`, we invoke `pyro.sample`:

```python
def model():
	inputs = pyro.sample('inputs', dist.Distribution(...))
	outputs = pyro.sample('outputs', dist.Empirical(
		numpy_simulator(my_simulator)(inputs)))
```

External simulators are often implemented as classes. Note that the class instance should be created outside of `model`: We will call `model` repeatedly inside the inference algorithm and do not want to create a new instance every time.

```python
my_simulator_instance = MyExternalSimulator(...)

def model():
	inputs = pyro.sample('inputs', dist.Distribution(...))
	outputs = pyro.sample('outputs', dist.Empirical(
		numpy_simulator(my_simulator_instance.forward)(inputs)))
```

 Note that our user carries the burden to:
- Understand that it's neccessary to define the simulator instance outside of the model function
- Take care of type conversion between NumPy and PyTorch
- Wrap the simulator outputs into an empirical distribution on which to call `pyro.sample`

Can we design an interface that implements the equivalent of the above while being more convenient/intuitive?

A "pyronic" way of doing this would be to use the same mechanisms that Pyro uses internally: use effect handlers and effectful functions (primitives).

This would allow the following equivalent way of specifying a model:

```python
@engine.simulator({'sim': MyExternalSimulator(...)})
def model():
	inputs = pyro.sample('inputs', dist.Distribution(...))
	outputs = engine.sample_simulator('outputs', 'sim', inputs)  # or: (inputs, 'sim')

# Note: The simulator is named in case we want to involve multiple simulators.
```

Alternative names for `engine.simulator` might be `engine.add_simulator`, or `engine.register_simulator`.

We could consider introducing `engine.sample` instead of `engine.sample_simulator`: When called with distributions as arguments, it would internally call `pyro.sample`, when called with reference to a simulator name, it would invoke the simulator:

```python
@engine.simulator({'sim': MyExternalSimulator(...)})
def model():
	inputs = engine.sample('inputs', dist.Distribution(...))
	outputs = engine.sample('outputs', 'sim', inputs)  # or: (inputs, 'sim')
```

Undecided whether this overloading is good or bad, but will stick with it for the next examples.


#### Offline simulators

The pattern from above can also be applied to cases where data has been collected offline: With an offline dataset we want to ensure that rows of dataset are always sampled together. We register the dataset and to our model, and `engine.sample` will internally take care to draw the output corresponding to the input:

```python
@engine.dataset({'inputs': torch.tensor([[...], [...]]), 'outputs': ...})
def model():
	inputs = engine.sample('inputs')
    outputs = engine.sample('outputs')
```

Potentially, I'm overloading `engine.sample` too heavily at this point. Can always use more primitives/arguments.


#### Hybrid cases

Hybrid cases are possible by adding multiple decorators:

```python
@engine.dataset({'inputs': ..., 'outputs': ...})
@engine.simulator({'sim': MyExternalSimulator(...)})
def model():
	# ...
```

#### Score compression

Score compression might make use of the same pattern, for instance by introducing
`engine.compress`:

```python
@engine.compressor({'comp': IMNN(...)})
@engine.simulator({'sim': MyExternalSimulator(...)})
def model():
	# ...
	outputs = engine.sample('outputs', 'sim', inputs)
	output_compressed = engine.compress('output_compressed', 'comp', output)
```


#### Observations

Models are conditioned on particular observed data for which we want
to do inference. For this we rely on the effect handler that is part of Pyro, e.g.,
using it as a decorator:

```python
@pyro.condition({'output': torch.tensor([[...]])})
def model():
	# ...
```

### Inference

#### Rejection ABC

The conditioned model is passed to the inference algorithm. We can undo the
conditioning inside the algorithm to create an unconditioned model to generate
samples from. (Alternatively, we pass an unconditioned model and the observations to an inference algorithm. It's also possible to write an interface allowing both.)

In the simplest case, an inference algorithm is a function that takes in
a model and returns a distribution (for instance a `dist.Empirical`):

```python
posterior = engine.RejectionABC(conditioned_model, ...)
```

For more complex cases we might have inference algorithms that are classes and have additional returns. E.g., we might want to be able to resume inference for additional steps and return diagnostics.


#### Density estimation based inference

For density estimation based algorithms, we'd additionally pass in density estimators and optimizers:

```python
density_estimator = ...  # nn.Module
posterior = engine.NeuralLikelihood(model, density_estimator, optims, ...)
posterior = engine.NeuralPosterior(model, density_estimator, optims, ...)
```

