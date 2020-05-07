# Modules on top of Pyro's poutine
from __future__ import absolute_import, division, print_function

import numpyro
from numpyro.primitives import Messenger

__all__ = ['simulator',
           'simulate']

class simulator(Messenger):
    """
    Inserts the effects of an online or offline simulator inside the execution
    of the model.
    """

    def __init__(self, name, simulator_fn=None, dataset=None):
        """
        :param name: Name of the simulator
        :param simulator_fn: Stochastic function defining the simulator
        :param dataset: Pyro dataloader instance yielding a dictionary of
                        simulated inputs and ouputs
        """
        super(simulator, self).__init__()
        self.name = name
        self.simulator_fn = simulator_fn
        self.dataset = dataset
        self.sample = None

        # Load the data buffer for the offline simulator
        if dataset is not None:
            self.dataset = enumerate(self.dataset)
            self._draw_offline_sample()

    def _draw_offline_sample(self):
        """
        Loads the next sample from the offline dataset
        """
        _, data = next(self.dataset)
        self.sample = data

    def process_message(self, msg):
        """
        :param msg: current message at a trace site
        """
        name = msg["name"]

        # Handle the simple case where we have an explicit simulator function
        if self.simulator_fn is not None:
            if name == self.name:
                msg['fn'] = self.simulator_fn

        # Otherwise, we use the offline dataset
        else:
            # If the current site name exists in the dataset override the value
            if name in self.sample:
                if msg['value'] is None:
                    msg['value'] = self.sample[name]

            # If we have reached the site of the simulator, also draw the next
            # sample
            if name == self.name:
                msg['fn'] = lambda *args, **kwargs: self.sample[name]
                self._draw_offline_sample()


def simulate(name,  *args, **kwargs):
    return numpyro.sample(name, None, *args, **kwargs)
