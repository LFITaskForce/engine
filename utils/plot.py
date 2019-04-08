import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import torch

from contextlib import contextmanager


@contextmanager
def mplrc():
    fname = os.path.join(os.path.dirname(__file__), 'matplotlibrc_dark')
    with mpl.rc_context(fname=fname):
        yield


def hist_1d(x, bins=25, title='', xlabel=''):
    x = x.detach().numpy() if type(x) == torch.Tensor else x

    with mplrc():
        plt.figure()
        plt.title(title)

        plt.hist(x, bins);

        plt.xlabel(xlabel)

        plt.show()
        plt.close()
