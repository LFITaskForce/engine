from benchmark.simulator_base import Simulator


class CythonSimulator(Simulator):
    def forward(self, inputs):
        raise NotImplementedError
