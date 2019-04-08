import pyro

from benchmark.simulator_base import Simulator


class PyroSimulator(Simulator):
    def forward(self, inputs):
        raise NotImplementedError

    def trace(self, inputs):
        return pyro.poutine.trace(self.forward).get_trace(inputs)
