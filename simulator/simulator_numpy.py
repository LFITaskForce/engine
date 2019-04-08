from benchmark.simulator_base import Simulator


class NumpySimulator(Simulator):
    def forward(self, inputs):
        raise NotImplementedError
