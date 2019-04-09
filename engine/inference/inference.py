from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed


class Inference(object):
    """Class for representing an inference method
    """

    def __init__(self, model, params, obs):
        """
        model: model function describing the generative process
        params: targets of inference
        obs: keywords for the parts of the model that are observed
        """
        self._model = model
        self._params = params
        self._obs = obs
