class Algorithm:
    def __init__(self, prior, simulator, obs, params={}):
        self.prior = prior
        self.simulator = simulator
        self.obs = obs
        self.params = params
