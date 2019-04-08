import tensorflow  # todo: import tfp

from benchmark.simulator_base import Simulator


class TFPSimulator(Simulator):
    def forward(self, inputs):
        raise NotImplementedError

    def trace(self, inputs):
        pass
