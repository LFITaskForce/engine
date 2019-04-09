from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
from .inference import Inference
from ..program_transformations import make_joint_smpl_fn


class RejectionABC(Inference):

    def __init__(self, model, params, obs,
                 distance_fn=lambda x, y: tf.norm(x - y, axis=-1),
                 epsilon=0.5):
        """ Simple rejection ABC algorithm
        """
        self._distance_fn = distance_fn
        self._epsilon = epsilon
        self._sampling_fn = make_joint_smpl_fn(model, params)
        super(self.__class__, self).__init__(model, params, obs)

    def run(self, n_samples):
        """
        Runs the inference
        """
        z, sims = self._sampling_fn(n_samples)
        discrepancies = self._distance_fn(sims, self._obs)
        mask = discrepancies < self._epsilon
        samples = ed.Empirical(tf.squeeze(sims[mask]))
        params_samples = {k: ed.Empirical(tf.squeeze(z[k][mask]),
                                          name='%s_posterior' % k) for k in z}
        return params_samples, samples
