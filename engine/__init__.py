__version__ = "0.0.1"

import pyro

_CTX = {}  # could use _PYRO_STACK (which is a list) instead

class SimulatorContext:
    def __init__(self, name, simulator):
        """
        :param name: string
        :param simulator: simulator instance
        """
        self.name = name
        self.simulator = simulator

    def __call__(self, func):
        def _wraps(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        _wraps.ctx = self
        return _wraps

    def __enter__(self):
        _CTX[self.name] = self.simulator

    def __exit__(self, exc_type, exc_value, traceback):
        del _CTX[self.name]


def register_simulator(fn=None, name=None, simulator=None):
    sim_dec = SimulatorContext(name=name, simulator=simulator)
    if fn is None:
        return sim_dec  # @register_simulator
    else:
        return sim_dec(fn)  # register_simulator(fn, ...)


def simulate(name, inputs):
    def fn():
        assert name in _CTX, 'simulator {} not registered'.format(name)
        return _CTX[name](inputs)
    return pyro.sample(name, fn)
