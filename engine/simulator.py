"""Classes and methods related to simulator definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six
from contextlib import contextmanager
from tensorflow_probability import edward2 as ed
from tensorflow_probability.python.internal import reparameterization
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow as tf
from tensorflow_probability.python.edward2.generated_random_variables import _make_random_variable

__all__ = ["OfflineSimulator"]

class _OfflineSimulator(tfd.Distribution):
    """
    This defines a placeholder for a simulator
    """
    def __init__(self, *input_args, sample_shape=(), name='simulator', dtype=tf.float32):
        super(self.__class__,self).__init__(dtype=dtype,
                                        validate_args=False,
                                        allow_nan_stats=True,
                                        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                                        name=name)


    def _sample_n(self, n, seed=None):
        raise RuntimeError("Provide the samples to the model.")

OfflineSimulator = _make_random_variable(_OfflineSimulator)
