import numpyro
import jax.numpy as np

from tqdm.auto import tqdm

class RejectionABC(object):
    """
    :param model: probabilistic model defined as a function
    :param float threshold: distance under which samples are accepted
    :param int num_samples: The number of samples that need to be generated,
        excluding the samples discarded during the warmup phase.
    :param int num_batch: The number of samples drawn in a batch.
    :param bool disable_progbar: Disable progress bar.

    Performs rejection ABC
    """

    def __init__(self, model, threshold, num_samples, disable_progbar=False, **static_kwargs):
        super(RejectionABC, self).__init__()
        #
        # trace = numpyro.trace(model).get_trace()
        # assert len(trace.observation_nodes) == 1, 'model should have a single observed site'

        self.static_kwargs = static_kwargs

        self.model = model
        self.threshold = threshold
        self.num_samples = num_samples
        self.disable_progbar = disable_progbar

    def run(self, rng_key, num_simulations=1, *args, **kwargs):
        """
        Initialize the mode
        """

        model_init = numpyro.seed(self.model, rng_key)
        model_trace = numpyro.trace(model_init).get_trace(*args, **kwargs)

        assert len(model_trace.observation_nodes) == 1, 'model should have a single observed site'
        obs_site = model_trace.observation_nodes[0]
        obs_value = model_trace.nodes[self.obs_site]['value']

        uncond_model = numpyro.seed(numpyro.uncondition(self.model), rng_key)
        uncond_trace = numpyro.trace(uncond_model) #.get_trace(*args, **kwargs)

        num_accepted = 0
        samples = []

        if not self.disable_progbar:
            pbar = tqdm(total=self.num_samples)
        while num_accepted < self.num_samples:
            trace = uncond_trace.get_trace(*args, **kwargs)
            discrepancy = torch.norm(
                trace.nodes[obs_site]['value'] - obs_value, dim=1)
            if discrepancy < self.threshold:
                num_accepted += 1
                if not self.disable_progbar:
                    pbar.update(1)
                samples.append(trace)

    #
    # def _sample_from_joint(self, *args, **kwargs):
    #     """
    #     :returns: a sample from the joint distribution over unobserved and
    #         observed variables
    #     :rtype: pyro.poutine.trace_struct.Trace
    #
    #     Returns a trace of the model without conditioning on any observations.
    #     """
    #     model = numpyro.uncondition(self.model)
    #     return numpyro.trace(model).get_trace(*args, **kwargs)
    #
    # def _traces(self, *args, **kwargs):
    #     num_accepted = 0
    #     if not self.disable_progbar:
    #         pbar = tqdm(total=self.num_samples)
    #     while num_accepted < self.num_samples:
    #         trace = self._sample_from_joint(*args, **kwargs)
    #         discrepancy = torch.norm(
    #             trace.nodes[self.obs_site]['value'] - self.obs_value, dim=1)
    #         if discrepancy < self.threshold:
    #             num_accepted += 1
    #             if not self.disable_progbar:
    #                 pbar.update(1)
    #             yield trace, 1.0
