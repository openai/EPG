import random
import time

import chainer.functions as F
import gym
import numpy as np


def reseed(env, pool_rank):
    np.random.seed(pool_rank + get_time_seed())
    random.seed(pool_rank + get_time_seed())
    env.seed(pool_rank + get_time_seed())


def sym_mean(x):
    return F.sum(x) / x.size


def gamma_expand(x, a):
    x, a = np.asarray(x), np.asarray(a)
    y = np.zeros_like(x)
    for t in reversed(range(len(x))):
        y[t] = x[t] + a[t] * (0 if t == len(x) - 1 else y[t + 1])
    return y


def get_dims(env):
    if isinstance(env.action_space, gym.spaces.Discrete):
        if isinstance(env.observation_space, gym.spaces.Discrete):
            env_dim = env.observation_space.n
        elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
            env_dim = env.observation_space.shape[0] * 3
        else:
            env_dim = env.observation_space.shape * 3
        act_dim = env.action_space.n
        n_output_params = 1
    else:
        env_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        n_output_params = 2
    return env_dim, act_dim, n_output_params


def int_to_onehot(x, dim):
    y = np.zeros(dim)
    y[x] = 1
    return y


def onehot_to_int(x):
    x = x.astype(int)
    return np.where(x == 1)[0][0]


def relative_ranks(x):
    def ranks(x):
        ranks = np.zeros(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    y = ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    return y / (x.size - 1.) - 0.5


class Adam(object):
    def __init__(self, shape, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08, dtype=np.float32):
        self.stepsize, self.beta1, self.beta2, self.epsilon = stepsize, beta1, beta2, epsilon
        self.t = 0
        self.m = np.zeros(shape, dtype=dtype)
        self.v = np.zeros(shape, dtype=dtype)

    def step(self, g):
        self.t += 1
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        return - a * self.m / (np.sqrt(self.v) + self.epsilon)


class Normalizer(object):
    def __init__(self, shape, epsilon=1e-2):
        self.shape = shape
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sum2 = np.full(shape, epsilon, dtype=np.float32)
        self.count = epsilon

    def _get_mean_and_std(self):
        mean = self.sum / self.count
        std = np.sqrt(np.maximum(self.sum2 / self.count - np.square(mean), 0.01))
        return mean, std

    def update(self, x):
        self.sum += np.sum(x, axis=0)
        self.sum2 += np.sum(np.square(x), axis=0)
        self.count += x.shape[0]

    def norm(self, x):
        mean, std = self._get_mean_and_std()
        return (x - mean) / std

    def unnorm(self, x):
        mean, std = self._get_mean_and_std()
        return mean + x * std


def gaussian_kl(params0, params1):
    (mean0, logstd0), (mean1, logstd1) = params0, params1
    assert mean0.shape == logstd0.shape == mean1.shape == logstd1.shape
    return F.sum(
        logstd1 - logstd0 + (F.square(F.exp(logstd0)) + F.square(mean0 - mean1)) / (
                2.0 * F.square(F.exp(logstd1))) - 0.5,
        axis=1
    )


def categorical_kl(params0, params1):
    params0 = params0[0]
    params1 = params1[0]
    assert params0.shape == params1.shape
    a0 = params0 - F.tile(F.max(params0, axis=1, keepdims=True), (1, 4))
    a1 = params1 - F.tile(F.max(params1, axis=1, keepdims=True), (1, 4))
    ea0 = F.exp(a0)
    ea1 = F.exp(a1)
    z0 = F.tile(F.sum(ea0, axis=1, keepdims=True), (1, 4))
    z1 = F.tile(F.sum(ea1, axis=1, keepdims=True), (1, 4))
    p0 = ea0 / z0
    return F.sum(p0 * (a0 - F.log(z0) - a1 + F.log(z1)), axis=1)


def log_misc_stats(key, logger, lst_value):
    lst_value = np.asarray(lst_value)
    logger.logkv(key + '~', np.mean(lst_value))
    logger.logkv(key + 'Median', np.median(lst_value))
    logger.logkv(key + 'Std', np.std(lst_value))
    logger.logkv(key + '-', np.min(lst_value))
    logger.logkv(key + '+', np.max(lst_value))


def get_time_seed():
    return int(1000000 * time.time() % 100000) * 1000


def ret_to_obj(ret):
    """Objective function
    """
    return np.mean(ret[-3:])


class Schedule(object):
    def value(self, t):
        raise NotImplementedError()


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        assert self._outside_value is not None
        return self._outside_value
