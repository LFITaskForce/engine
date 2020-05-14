# Modules on top of Pyro's poutine
from __future__ import absolute_import, division, print_function

import numpyro as pyro
# import pyro
from numpyro.handlers import Messenger
# from pyro.poutine.messenger import Messenger

__all__ = ['simulator', 'simulate', 'uncondition']


class SimulatorMessenger(Messenger):
    """
    Inserts the effects of an online or offline simulator inside the execution
    of the model.
    """

    def __init__(self, name, model_fn, simulator_fn=None, dataset=None):
        """
        :param name: Name of the simulator
        :param simulator_fn: Stochastic function defining the simulator
        :param dataset: Pyro dataloader instance yielding a dictionary of
                        simulated inputs and ouputs
        """
        super().__init__(model_fn)
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

    def _pyro_sample(self, msg):
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


class uncondition(Messenger):
    """
    Messenger to force the value of observed nodes to be sampled from their
    distribution, ignoring observations.
    """
    def __init__(self, fn=None):
        super().__init__(fn)

    def process_message(self, msg):
        """
        :param msg: current message at a trace site.

        Samples value from distribution, irrespective of whether or not the
        node has an observed value.
        """
        if msg["is_observed"]:
            msg["is_observed"] = False
            # msg["infer"]["was_observed"] = True
            msg["obs"] = msg["value"]
            msg["value"] = None
            msg["done"] = False
        return None


def simulator(name, simulator_fn=None, dataset=None):
    return lambda fn: SimulatorMessenger(name, fn, simulator_fn, dataset)


def simulate(name, *args, **kwargs):
    return pyro.sample(name, *args, **kwargs)
