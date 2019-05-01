from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

import tensorflow_probability as tfp
import tqdm
import numpy as np

class IMNN(object):
    """Class for representing IMNN compression
    """

    def __init__(self, model, fiducial):
        """
        model: model architecture for IMNN
        fiducial: array of fiducial parameter values
        """
        self._model = model
        self.fiducial = fiducial

        self.history = {"F": [], "F_val": []}
        self.dddt_train = None
        self.dddt_val = None
        self.dm_set = None
        self.dp_set = None
        self.dm_val = None
        self.dp_val = None
        self.train_d_ind = None
        self.train_ind = None
        self.n_s = None
        self.n_p = None
        self.sets = None

    ''' Could potentially set up datasets
    def datasets(self, n_s, d, dm=None, dp=None, dt=None, dddt=None,
                 train_set=None, val_set=None, n_train=None, numerical=None):
        """
        n_s: number of simulations to calculate covariance with
        d: fiducial dataset to fit IMNN
        dm: dataset at lower derivative point to calculate derivative
        dp: dataset at upper derivative point to calculate derivative
        dt: array of parameter differences to calculate derivative
        dddt: dataset of derivative of data with respect to parameters
        """
        if (dm is not None) and (dp is not None) and (dt is not None):
            train_set = (d[:-n_s], d_m[:-n_s], d_p[:-n_s])
            n_train = train_set[0].shape[0]
            val_set = (d[-n_s:], d_m[-n_s:], d_p[-n_s:])
            self._dt = tf.get_variable(
                "dt",
                shape=(self._n_params),
                dtype=tf.float32,
                initializer=tf.constant_initializer(dt))
            self._numerical = True
        elif dddt is not None:
            train_set = dddt[:-n_s]
            n_train = train_set[0].shape[0]
            val_set = dddt[-n_s:]
            self._numerical = False
        elif (train_set is not None) and (val_set is not None) \
                and (n_train is not None) and (numerical is not None):
            _train_set = train_set
            n_train = n_train
            _val_set = val_set
            self._numerical = numerical
        else:
            raise

        if ((dm is not None) and (dp is not None) and (dt is not None)) \
                or (dddt is not None):
                _train_set = tf.data.Dataset.from_tensor_slices(train_set)
                _val_set = tf.data.Dataset.from_tensor_slices(val_set)
        d_train = _train_set.cache().shuffle(n_train)\
            .batch(n_s).prefetch(buffer_size=n_train)
        d_val = _val_set.cache().batch(n_s).prefetch(buffer_size=n_s)
        iterator = tf.data.Iterator.from_structure(d_train.output_types,
                                                   d_train.output_shapes)
        if self._numerical:
            self._d, self._dm, self._dp = iterator.get_next()
        else:
            self._d, self._dddt = iterator.get_next()
    '''

    def covariance(self, x):
        """ Calculate the covariance of the summaries
        x: summary from the model
        """
        m = tf.reduce_mean(x, axis=0, keepdims=True)
        mx = tf.matmul(tf.transpose(m), m)
        denominator = tf.cast(tf.shape(x)[0] - 1, tf.float32)
        vx = tf.divide(tf.matmul(tf.transpose(x), x), denominator)
        return tf.subtract(vx, mx)

    def sim_based_mean(self, d, dddt):
        """ Calculate derivative of summaries with respect to the parameters
        d: input data at fiducial value
        dddt: input derivative of the data with respect to the parameters
        """
        with tf.GradientTape() as tape:
            tape.watch(d)
            x = self._model(d)
        dxdd = tape.batch_jacobian(x, d)
        inter_indices = tf.range(2, tf.rank(network_derivative))
        transpose_indices = tf.concat([[0], inter_indices, [1]], 0)
        shaped_derivative = tf.transpose(dxdd, transpose_indices)
        dxdt = tf.matmul(dddt, shaped_derivative)
        dmdfdt = tf.reduce_mean(dxdt, axis=0)
        return x, dmfdt

    def numerical_mean(self, dp, dm, dt):
        """ Calculate derivative of summaries with respect to the parameters
        dp: input data at upper derivative point for numerical derivative
        dm: input data at lower derivative point for numerical derivative
        dt: parameter differences to calculate numerical derivative
        """
        xp = self._model(dp)
        xm = self._model(dm)
        dxdt = tf.divide(tf.subtract(xp, xm), dt)
        return tf.reduce_mean(dxdt, axis=0)

    def get_F_components(self, d, dddt=None, dp=None, dm=None, dt=None):
        """ Collects summaries, derivatives of the mean and covariance
        d: input data at fiducial value
        dddt: input derivative of the data with respect to the parameters
        dp: input data at upper derivative point for numerical derivative
        dm: input data at lower derivative point for numerical derivative
        dt: parameter differences to calculate numerical derivative
        """
        if self.numerical:
            x = self._model(d)
            dmfdt = self.numerical_mean(dp, dm, dt)
        else:
            x, dmfdt = self.sim_based_mean(d, dddt)
        return x, dmfdt, self.covariance(x)

    def Fisher(self, dmdt, C):
        """ Calculate Fisher matrix
        dmdt: derivative of the mean of the summaries
        C: covariance of the summaries
        """
        invC = tf.linalg.inv(C)
        return tf.matmul(tf.matmul(dmdt, invC), tf.transpose(dmdt))

    def loss(self, F, C, lam):
        """ Calculate the negative log determinant of the Fisher + constraint
        F: Fisher information matrix
        C: covariance of the summaries
        lam: constraint strength for setting scale of the summaries
        """
        lndetF = tf.linalg.slogdet(F)
        gathered_lndetF = tf.multiply(lndetF[0], lndetF[1])
        cons = tf.subtract(C, tf.eye(tf.shape(C)[0], dtype=tf.float32))
        cons_norm = tf.norm(cons, ord="euclidean")
        scaled_cons = tf.multiply(lam, cons_norm)
        return tf.subtract(scaled_cons, gathered_lndetF)

    def get_Fisher(self, d, dddt=None, dp=None, dm=None, dt=None,
                   return_C=False, return_detF=False):
        """ Wrapper to calculate Fisher information straight from data
        d: input data at fiducial value
        dddt: input derivative of the data with respect to the parameters
        dp: input data at upper derivative point for numerical derivative
        dm: input data at lower derivative point for numerical derivative
        dt: parameter differences to calculate numerical derivative
        """
        x, dmfdt, Cf = self.get_F_components(d, dddt=dddt, dp=dp, dm=dm, dt=dt)
        F = self.Fisher(dmfdt, Cf)
        if return_C:
            return F, Cf
        elif return_detF:
            return tf.linalg.det(F)
        else:
            return F

    def train_step(self, d, lam, optimiser, dm=None, dp=None, dt=None,
                   dddt=None):
        """ Does a single weight update of the model
        d: input data at fiducial value
        lam: constraint strength for setting scale of the summaries
        optimiser: optimisation scheme for updating the network
        dddt: input derivative of the data with respect to the parameters
        dp: input data at upper derivative point for numerical derivative
        dm: input data at lower derivative point for numerical derivative
        dt: parameter differences to calculate numerical derivative
        """
        with tf.GradientTape() as g:
            F, Cf = self.get_Fisher(d, dp=dp, dm=dm, dt=dt, dddt=dddt,
                                    return_C=True)
            L = self.loss(F, Cf, lam)
        gradients = g.gradient(L, self._model.variables)
        optimiser.apply_gradients(zip(gradients, self._model.variables))
        return tf.linalg.det(F)

    def summary(self, d):
        """ Returns an IMNN summary
        d: input data
        """
        return self._model(d)

    def MLE(self, d):
        """ Returns a maximum likelihood estimate
        d: input data
        fiducial: array of the fiducial parameter values
        """
        diff = tf.subtract(self.summary(d), self.mean)
        c_diff = tf.einsum("ij,kj->ki", self.compressor, diff)
        c = tf.einsum("ij,kj->ki", self.invF, c_diff)
        return tf.add(self.fiducial, c)

    def fit(self, optimiser, lam, updates=1000):
        """ Fit the IMNN to give optimal summaries
        d: input data at fiducial value
        n_s: number of simulations to estimate covariance
        optimiser: optimisation scheme for updating the network
        lam: constraint strength for setting scale of the summaries
        updates: number of passes through entire data for training
        """

        bar = tqdm.trange(updates, desc="Updates")
        for update in bar:
            # IDEALLY I WOULD MAKE THE DATA A TF.DATA.DATASET() AND DO
            # EVERYTHING IN TENSORFLOW BUT PASSING NUMPY ARRAYS SEEMS TO BE
            # MORE TF2...?
            np.random.shuffle(self.train_ind)
            for this_set in range(self.sets):
                start_ind = this_set * self.n_s
                end_ind = start_ind + self.n_s
                set_ind = self.train_ind[start_ind: end_ind]
                if self.numerical:
                    np.random.shuffle(self.train_d_ind)
                    set_d_ind = self.train_d_ind[:self.n_p]
                    dm_set = self.dm_train[set_d_ind]
                    dp_set = self.dp_train[set_d_ind]
                    dddt_set = None
                else:
                    dddt_set = self.dddt[set_ind]
                    dm_set, dp_set = None, None
                temp_F = self.train_step(self.d_train[set_ind], lam, optimiser,
                                         dm=dm_set, dp=dp_set, dt=self.dt,
                                         dddt=dddt_set)
            self.history["F"].append(temp_F)
            self.history["F_val"].append(
                self.get_Fisher(self.d_val, dp=self.dp_val, dm=self.dm_val,
                                dt=self.dt, dddt=self.dddt_val,
                                return_detF=True))
            bar.set_postfix(F=self.history["F"][-1],
                            val_F=self.history["F_val"][-1])

        summaries, self.dmfdt, self.C = \
            self.get_F_components(self.d_val, dddt=self.dddt_val,
                                  dp=self.dp_val, dm=self.dm_val, dt=self.dt)
        self.mean = tf.reduce_mean(summaries, axis=0)
        self.F = self.Fisher(self.dmfdt, self.C)
        self.invF = tf.linalg.inv(self.F)
        self.forecast = tf.sqrt(tf.linalg.tensor_diag(self.invF))
        self.compressor = tf.einsum("ij,kj->ki",
                                    tf.linalg.inv(self.C),
                                    self.dmfdt)

    def load_dataset(self, n_s, d, dm=None, dp=None, dt=None, dddt=None,
                     n_p=None):
        self.d_train = d[:-n_s]
        self.d_val = d[-n_s:]
        n_train = self.d_train.shape[0]
        self.train_ind = np.arange(n_train)
        self.n_s = n_s
        self.sets = int(n_train / n_s)
        if (dm is not None) and (dp is not None) and (dt is not None) \
                and (n_p is not None):
            self.numerical = True
            self.dt = dt
            self.dm_train = dm[:-n_p]
            self.dm_val = dm[-n_p:]
            self.dp_train = dp[:-n_p]
            self.dp_val = dp[-n_p:]
            self.train_d_ind = np.arange(self.dm_train.shape[0])
            self.n_p = n_p
        elif (dddt is not None):
            self.numerical = False
            self.dddt_train = dddt[:-n_s]
            self.dddt_val = dddt[-n_s:]

    def generate_dataset(self, simulator, n_s, n_p, n_fid, n_der, h=None):
        """
        simulator: simulator with batch_simulate with or without derivatives
        n_s: number of simulations needed to accurate estimate of covariance
        n_p: number of simulations needed to get accurate estimate of mean
        n_fid: number of simulations to make at fiducial value (> n_s)
        n_der: number of simulations to make for numerical derivative (> n_p)
        h: distance from fiducial to make simulations for numerical derivative
        """
        self.train_ind = np.arange(n_fid)
        self.n_s = n_s
        self.sets = int(n_fid / n_s)
        if h is None:
            self.numerical = False
            self.d_train, self.dddt_train = \
                simulator.batch_simulate(self.fiducial, n_fid)
            self.d_train = self.d_train.astype(np.float32)
            self.dddt_train = self.dddt_train.astype(np.float32)
            self.d_val, self.dddt_val = \
                simulator.batch_simulate(self.fiducial, n_s)
            self.d_test = self.d_test.astype(np.float32)
            self.dddt_test = self.dddt_test.astype(np.float32)
        if h is not None:
            self.numerical = True
            self.dt = 2. * h
            self.d_train = simulator.batch_simulate(
                self.fiducial, n_fid).astype(np.float32)
            self.d_val = simulator.batch_simulate(
                self.fiducial, n_s).astype(np.float32)
            self.dp_train, self.dm_train = \
                simulator.batch_get_derivatives(self.fiducial, h, n_der)
            self.dp_train = self.dp_train.astype(np.float32)
            self.dm_train = self.dm_train.astype(np.float32)
            self.dp_val, self.dm_val = \
                simulator.batch_get_derivatives(self.fiducial, h, n_p)
            self.dp_val = self.dp_val.astype(np.float32)
            self.dm_val = self.dm_val.astype(np.float32)
            self.train_d_ind = np.arange(n_der)
