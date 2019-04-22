import tensorflow as tf

__all__ = ["model"]

def model(_func=None):
    """
    Decorator that turns a model function into an actual model

    Parameters
    ----------

    Returns
    -------
    The model defined by the function
    """

    def wrap(model_fn):
        return Model(model_fn)

    if _func is None:
        return wrap
    else:
        return wrap(_func)


class Model:
    """
    Class used to store the probabilistic model
    """
    def __init__(self, model_fn):
        self.model_fn = model_fn

    def __call__(self, *args, **kwargs):
        return self.model_fn(*args, **kwargs)
