# Design document for LFI Engine

## Objective

Provide a simple and efficient framework for performing Likelihood-Free Inference
on top of a probabilistic programing language.

## Related work

Related projects can be broken down in two classes, generic inference frameworks
and LFI implementations.

Inference frameworks:
  - [PyMC3](https://docs.pymc.io/) provides both intuitive syntax for defining a
  model, and set of inference algorithms (NUTS, ADVI). Theano based.
  - [Pyro](http://pyro.ai/)

TODO LFI engines:


## Design overview

1. Framework should remain lightweight, sticking as much as possible to existing
  conventions and tools from most closely related back-end PPL.

### Example 1: Simple ABC on a given model



## Detailed design
