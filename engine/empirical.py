# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import numpy as np
from numpyro.distributions import constraints
from numpyro.distributions import Categorical
from numpyro.distributions import Distribution


class Empirical(Distribution):
    r"""
    Empirical distribution associated with the sampled data. Note that the shape
    requirement for `log_weights` is that its shape must match the leftmost shape
    of `samples`. Samples are aggregated along the ``aggregation_dim``, which is
    the rightmost dim of `log_weights`.

    Example:

    >>> emp_dist = Empirical(np.random.randn(2, 3, 10), np.ones(2, 3))
    >>> emp_dist.batch_shape
    (2,)
    >>> emp_dist.event_shape
    (10,)

    >>> single_sample = emp_dist.sample()
    >>> single_sample.shape
    (2, 10)
    >>> batch_sample = emp_dist.sample((100,))
    >>> batch_sample.shape
    (100, 2, 10)

    >>> emp_dist.log_prob(single_sample).shape
    (2,)
    >>> # Vectorized samples cannot be scored by log_prob.
    >>> with pyro.validation_enabled():
    ...     emp_dist.log_prob(batch_sample).shape
    Traceback (most recent call last):
    ...
    ValueError: ``value.shape`` must be (2, 10)

    :param np.array samples: samples from the empirical distribution.
    :param np.array log_weights: log weights (optional) corresponding
        to the samples.
    """

    arg_constraints = {}
    support = constraints.real
    has_enumerate_support = True

    def __init__(self, samples, log_weights, validate_args=None):
        self._samples = samples
        self._log_weights = log_weights
        sample_shape, weight_shape = samples.shape, log_weights.shape
        if weight_shape > sample_shape or weight_shape != sample_shape[:len(weight_shape)]:
            raise ValueError("The shape of ``log_weights`` ({}) must match "
                             "the leftmost shape of ``samples`` ({})".format(weight_shape, sample_shape))
        self._aggregation_dim = len(log_weights) - 1
        event_shape = sample_shape[len(weight_shape):]
        self._categorical = Categorical(logits=self._log_weights)
        super().__init__(batch_shape=weight_shape[:-1],
                         event_shape=event_shape,
                         validate_args=validate_args)

    @property
    def sample_size(self):
        """
        Number of samples that constitute the empirical distribution.

        :return int: number of samples collected.
        """
        return self._log_weights.size

    def sample(self, sample_shape=()):
        sample_idx = self._categorical.sample(sample_shape)  # sample_shape x batch_shape
        # reorder samples to bring aggregation_dim to the front:
        # batch_shape x num_samples x event_shape -> num_samples x batch_shape x event_shape
        samples = self._samples.unsqueeze(0).transpose(0, self._aggregation_dim + 1).squeeze(self._aggregation_dim + 1)
        # make sample_idx.shape compatible with samples.shape: sample_shape_numel x batch_shape x event_shape
        sample_idx = sample_idx.reshape((-1,) + self.batch_shape + (1,) * len(self.event_shape))
        sample_idx = sample_idx.expand((-1,) + samples.shape[1:])
        return samples.gather(0, sample_idx).reshape(sample_shape + samples.shape[1:])

    def log_prob(self, value):
        """
        Returns the log of the probability mass function evaluated at ``value``.
        Note that this currently only supports scoring values with empty
        ``sample_shape``.

        :param np.array value: scalar or tensor value to be scored.
        """
        if self._validate_args:
            if value.shape != self.batch_shape + self.event_shape:
                raise ValueError("``value.shape`` must be {}".format(self.batch_shape + self.event_shape))
        if self.batch_shape:
            value = value.unsqueeze(self._aggregation_dim)
        selection_mask = self._samples.eq(value)
        # Get a mask for all entries in the ``weights`` tensor
        # that correspond to ``value``.
        for _ in range(len(self.event_shape)):
            selection_mask = selection_mask.min(axis=-1)[0]
        selection_mask = selection_mask.type(self._categorical.probs.type())
        return np.log(np.sum(self._categorical.probs * selection_mask, axis=-1))

    def _weighted_mean(self, value, keepdims=False):
        weights = self._log_weights.reshape(self._log_weights.shape +
                                            ([1] * (len(value) - len(self._log_weights)),))
        dim = self._aggregation_dim
        max_weight = weights.max(axis=dim, keepdim=True)[0]
        relative_probs = np.exp(weights - max_weight)
        return np.sum(value * relative_probs, axis=dim, keepdims=keepdims) \
            / np.sum(relative_probs, axis=dim, keepdims=keepdims)

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def mean(self):
        if self._samples.dtype in (np.int32, np.int64):
            raise ValueError("Mean for discrete empirical distribution undefined. " +
                             "Consider converting samples to ``np.float32`` " +
                             "or ``np.float64``. If these are samples from a " +
                             "`Categorical` distribution, consider converting to a " +
                             "`OneHotCategorical` distribution.")
        return self._weighted_mean(self._samples)

    @property
    def variance(self):
        if self._samples.dtype in (np.int32, np.int64):
            raise ValueError("Variance for discrete empirical distribution undefined. " +
                             "Consider converting samples to ``np.float32`` " +
                             "or ``np.float64``. If these are samples from a " +
                             "`Categorical` distribution, consider converting to a " +
                             "`OneHotCategorical` distribution.")
        mean = self.mean.unsqueeze(self._aggregation_dim)
        deviation_squared = np.pow(self._samples - mean, 2)
        return self._weighted_mean(deviation_squared)

    @property
    def log_weights(self):
        return self._log_weights

    def enumerate_support(self, expand=True):
        # Empirical does not support batching, so expanding is a no-op.
        return self._samples
