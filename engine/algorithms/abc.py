import numpy as np
import numpyro
from numpyro import handlers as poutine
# import pyro.poutine as poutine
# import torch

from engine.gravy import uncondition

from pyro.infer.abstract_infer import TracePosterior
from tqdm.auto import tqdm


class RejectionABC(TracePosterior):
    """
    :param model: probabilistic model defined as a function
    :param float threshold: distance under which samples are accepted
    :param int num_samples: The number of samples that need to be generated,
        excluding the samples discarded during the warmup phase.
    :param int num_batch: The number of samples drawn in a batch.
    :param bool disable_progbar: Disable progress bar.

    Performs rejection ABC
    """

    def __init__(self, model, threshold, num_samples, disable_progbar=False):
        super().__init__()
        # super(RejectionABC, self).__init__()

        trace = poutine.trace(model).get_trace()
        assert len(trace) == 1, 'model should have a single observed site, but got: {}'.format(trace.keys())

        self.model = model
        self.threshold = threshold
        self.num_samples = num_samples
        self.disable_progbar = disable_progbar

        self.obs_site = list(trace.keys())[0]
        self.obs_value = trace[self.obs_site]['value']

    def _sample_from_joint(self, *args, **kwargs):
        """
        :returns: a sample from the joint distribution over unobserved and
            observed variables
        :rtype: pyro.poutine.trace_struct.Trace

        Returns a trace of the model without conditioning on any observations.
        """
        model = uncondition(self.model)
        return poutine.trace(model).get_trace(*args, **kwargs)

    def _traces(self, *args, **kwargs):
        num_accepted = 0
        if not self.disable_progbar:
            pbar = tqdm(total=self.num_samples)
        while num_accepted < self.num_samples:
            trace = self._sample_from_joint(*args, **kwargs)
            discrepancy = np.linalg.norm(trace[self.obs_site]['value'] - self.obs_value)
            if discrepancy < self.threshold:
                num_accepted += 1
                if not self.disable_progbar:
                    pbar.update(1)
                yield trace, 1.0
