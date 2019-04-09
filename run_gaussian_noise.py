import numpy as np
import matplotlib.pyplot as plt
from engine.inference.abc import RejectionABC
from tensorflow_probability import edward2 as ed

if __name__ == "__main__":

    # Define the model as an Edward function
    def model(num_samples):
        # Define priors
        mu = ed.Normal(loc=0., scale=1., name='mu', sample_shape=(num_samples, 1))
        # Simulator
        obs = ed.Normal(loc=mu, scale=0.1, name='obs')
        # Return the observations
        return obs

    # Create mock observation
    obs = np.array([[1.]])

    # Paramaters over which to run the inference
    params = ['mu']

    # Instantiate inference engine
    inf_engine = RejectionABC(model, params, obs, epsilon=0.01)

    # Returns empirical posterior distribution, along with valid samples
    posterior, samples = inf_engine.run(n_samples=1000000)

    # Let's extract a few samples from the posterior for plotting
    samps = posterior['mu'].distribution.sample(100000)

    plt.hist(samps, 64, label='Posterior Samples')
    plt.axvline(1, color='red', label='truth')
    plt.show()
