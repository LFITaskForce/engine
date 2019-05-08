from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import six
import tensorflow as tf

from tensorflow_probability import edward2 as ed

__all__ = [
    "make_joint_smpl_fn"
]

def make_joint_smpl_fn(model, targets):
    """Takes an Edward probabilistic model and returns samples from the joint
    distribution of specified target parameters and observables

    Args:
      model: Python callable which executes the generative process of a
      computable probability distribution using `ed.RandomVariable`s.

      targets: list of names of variables to save during prgram execution

    Returns:
      function that generates samples from (theta, y)
    """
    def sampling_fn(*args):
        with ed.tape() as model_tape:
            obs = model(*args)
        # Clean up and remove unwanted variables from tape
        # for i in model_tape:
        #     if i not in targets:
        #         model_tape.pop(i)
        return model_tape, obs
    return sampling_fn
