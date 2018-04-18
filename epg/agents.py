import chainer as C
import chainer.functions as F
import numpy as np
from mpi4py import MPI

from epg.launching import logger
from epg.exploration import HashingBonusEvaluator
from epg.losses import Conv1DLoss
from epg.networks import NN, Memory
from epg.utils import sym_mean, gamma_expand, int_to_onehot, onehot_to_int, \
    Adam, Normalizer, gaussian_kl, categorical_kl


class GenericAgent(object):
    def __init__(self,
                 env_dim, act_dim, policy_output_params,
                 memory_out_size=None, inner_n_opt_steps=None, inner_opt_batch_size=None,
                 inner_use_ppo=None, mem=None, buffer_size=None):
        assert inner_n_opt_steps is not None
        assert inner_opt_batch_size is not None
        assert inner_use_ppo is not None
        self._use_ppo = inner_use_ppo
        if self._use_ppo:
            self._ppo_gam = 0.99
            self._ppo_lam = 0.95
            self._ppo_klcoeff = 0.001
            self._ppo_clipparam = 0.2
            self._vf = NN([env_dim] + list([64, 64]) + [1], out_fn=lambda x: x)

        self.pi = None
        self._logstd = None
        self._use_mem = mem
        self._buffer_size=buffer_size

        self.inner_n_opt_steps = inner_n_opt_steps
        self.inner_opt_batch_size = inner_opt_batch_size

        self._mem_out_size = memory_out_size
        self._mem = Memory(64, self._mem_out_size)

        self.lst_rew_bonus_eval = [HashingBonusEvaluator(dim_key=128, obs_processed_flat_dim=env_dim)]

        self._env_dim = env_dim
        self._act_dim = act_dim

        # obs_dim, act_dim, rew, aux, done, pi params
        self._traj_in_dim = env_dim + act_dim + len(
            self.lst_rew_bonus_eval) + 2 + policy_output_params * act_dim + self._mem_out_size

        self._loss = Conv1DLoss(traj_dim_in=self._traj_in_dim)
        self._traj_norm = Normalizer((env_dim + act_dim + len(self.lst_rew_bonus_eval) + 2,))

    @property
    def backprop_params(self):
        if self._use_mem:
            if self._use_ppo:
                return self._vf.train_vars + self._mem.train_vars
            else:
                return self._mem.train_vars
        else:
            if self._use_ppo:
                return self._vf.train_vars
            else:
                return []

    def _pi_f(self, x):
        raise NotImplementedError

    def _pi_logp(self, x, y):
        raise NotImplementedError

    def kl(self, params0, params1):
        raise NotImplementedError

    def _logp(self, params, acts):
        raise NotImplementedError

    def _vf_f(self, x):
        return self._vf.f(x)

    def act(self, obs):
        raise NotImplementedError

    def set_loss(self, loss):
        self._loss = loss

    def get_loss(self):
        return self._loss

    def _process_trajectory(self, traj):
        proc_traj_in = F.concat(
            [traj] + self._pi_f(traj[..., :self._env_dim]) + \
            [F.tile(self._mem.f(), (traj.shape[0], 1)).data],
            axis=1
        )
        return self._loss.process_trajectory(proc_traj_in)

    def _compute_loss(self, traj, processed_traj):
        loss_inputs = [traj, processed_traj] + \
                      self._pi_f(traj[..., :self._env_dim]) + \
                      [F.tile(self._mem.f(), (traj.shape[0], 1))]
        loss_inputs = F.concat(loss_inputs, axis=1)
        epg_surr_loss = self._loss.loss(loss_inputs)
        return epg_surr_loss

    def _compute_ppo_loss(self, obs, acts, at, vt, old_params):
        params = self._pi_f(obs)
        cv = F.flatten(self._vf_f(obs))
        ratio = F.exp(self._logp(params, acts) - self._logp(old_params, acts))
        surr1 = ratio * at
        surr2 = F.clip(ratio, 1 - self._ppo_clipparam, 1 + self._ppo_clipparam) * at
        ppo_surr_loss = (
                -sym_mean(F.minimum(surr1, surr2))
                + self._ppo_klcoeff * sym_mean(self.kl(old_params, params))
                + sym_mean(F.square(cv - vt))
        )
        return ppo_surr_loss

    def update(self, obs, acts, rews, dones, ppo_factor, inner_opt_freq):

        epg_rews = rews
        # Want to zero out rewards to the EPG loss function?
        # epg_rews = np.zeros_like(rews)

        # Calculate auxiliary functions.
        lst_bonus = []
        for rew_bonus_eval in self.lst_rew_bonus_eval:
            lst_bonus.append(rew_bonus_eval.predict(obs).T)
        auxs = np.concatenate(lst_bonus, axis=0)

        traj_raw = np.c_[obs, acts, epg_rews, auxs, dones].astype(np.float32)
        # Update here, since we only have access to these raws at this specific spot.
        self._traj_norm.update(traj_raw)
        traj = self._traj_norm.norm(traj_raw)
        auxs_pad = np.zeros(self._buffer_size - obs.shape[0], dtype=np.float32)
        rew_pad = np.zeros(self._buffer_size - obs.shape[0], dtype=np.float32)
        done_pad = np.zeros(self._buffer_size - obs.shape[0], dtype=np.float32)
        obs_pad = np.zeros((self._buffer_size - obs.shape[0], obs.shape[1]), dtype=np.float32)
        act_pad = np.zeros((self._buffer_size - acts.shape[0], acts.shape[1]), dtype=np.float32)
        pad = np.hstack([obs_pad, act_pad, rew_pad[:, None], auxs_pad[:, None], done_pad[:, None]])
        traj = np.vstack([pad, traj])
        traj[:, obs.shape[1] + acts.shape[1]] = epg_rews
        traj[:, -1] = dones

        # Since the buffer length can be larger than the set of new samples, we truncate the
        # trajectories here for PPO.
        dones = dones[-inner_opt_freq:]
        rews = rews[-inner_opt_freq:]
        acts = acts[-inner_opt_freq:]
        obs = obs[-inner_opt_freq:]
        _obs = traj[-inner_opt_freq:, :obs.shape[1]]
        n = len(obs)

        if self._use_ppo:
            old_params_sym = self._pi_f(_obs)
            vp = np.ravel(self._vf_f(_obs).data)
            old_params = [item.data for item in old_params_sym]
            advs = gamma_expand(rews + self._ppo_gam * (1 - dones) * np.append(vp[1:], vp[-1]) - vp,
                                self._ppo_gam * self._ppo_lam * (1 - dones))
            vt = advs + vp
            at = (advs - advs.mean()) / advs.std()

        epg_surr_loss = 0.
        pi_params_before = self._pi_f(_obs)
        for _ in range(self.inner_n_opt_steps):
            for idx in np.array_split(np.random.permutation(n), n // self.inner_opt_batch_size):
                # Clear gradients
                for v in self.backprop_params:
                    v.cleargrad()

                # Forward pass through loss function.
                # Apply temporal conv to input trajectory
                processed_traj = self._process_trajectory(traj)
                # Compute epg loss value
                epg_surr_loss_sym = self._compute_loss(traj[idx], processed_traj[idx])
                epg_surr_loss += epg_surr_loss_sym.data

                # Add bootstrapping signal if needed.
                if self._use_ppo:
                    old_params_idx = [item[idx] for item in old_params]
                    ppo_surr_loss = self._compute_ppo_loss(
                        _obs[idx], acts[idx], at[idx], vt[idx], old_params_idx)
                    total_surr_loss = epg_surr_loss_sym * (1 - ppo_factor) + ppo_surr_loss * ppo_factor
                else:
                    total_surr_loss = epg_surr_loss_sym

                # Backward pass through loss function
                total_surr_loss.backward()
                for v, adam in zip(self.backprop_params, self._lst_adam):
                    if np.isnan(v.grad).any() or np.isinf(v.grad).any():
                        logger.log(
                            "WARNING: gradient update nan on node {}".format(MPI.COMM_WORLD.Get_rank()))
                    else:
                        v.data += adam.step(v.grad)

        pi_params_after = self._pi_f(_obs)

        return epg_surr_loss / (n // self.inner_opt_batch_size) / self.inner_n_opt_steps, \
               np.mean(self.kl(pi_params_before, pi_params_after).data)


class ContinuousGenericAgent(GenericAgent):
    def __init__(self, env_dim, act_dim, inner_lr=None, **kwargs):
        assert inner_lr is not None
        super().__init__(env_dim, act_dim, 2, **kwargs)
        self._pi = NN([env_dim] + list([64, 64]) + [act_dim], out_fn=lambda x: x)
        self._logstd = C.Variable(np.zeros(act_dim, dtype=np.float32))
        self._lst_adam = [Adam(var.shape, stepsize=inner_lr) for var in self.backprop_params]

    @property
    def backprop_params(self):
        return super(ContinuousGenericAgent, self).backprop_params + self._pi.train_vars + [self._logstd]

    def _pi_f(self, x):
        return [self._pi.f(x), F.tile(self._logstd, (x.shape[0], 1))]

    def _pi_logp(self, obs, acts):
        mean, logstd = self._pi_f(obs)
        return (
                - 0.5 * np.log(2.0 * np.pi) * acts.shape[1]
                - 0.5 * F.sum(F.square((acts - mean) / (F.exp(logstd)) + 1e-8), axis=1)
                - F.sum(logstd, axis=1)
        )

    def _logp(self, params, acts):
        mean, logstd = params
        return (
                - 0.5 * np.log(2.0 * np.pi) * acts.shape[1]
                - 0.5 * F.sum(F.square((acts - mean) / (F.exp(logstd)) + 1e-8), axis=1)
                - F.sum(logstd, axis=1)
        )

    def act(self, obs):
        obs = obs.astype(np.float32)
        # Use same normalization as traj.
        traj = np.concatenate(
            [obs, np.zeros(self._act_dim + 2 + len(self.lst_rew_bonus_eval), dtype=np.float32)])
        # Normalize!
        obs = self._traj_norm.norm(traj)[:self._env_dim]
        mean = self._pi.f(obs[np.newaxis, ...]).data
        std = np.exp(self._logstd.data)
        assert (std > 0).all(), 'std not > 0: {}'.format(std)
        return np.random.normal(loc=mean, scale=std).astype(np.float32)[0]

    def kl(self, params0, params1):
        return gaussian_kl(params0, params1)

    @staticmethod
    def act_to_env_format(act):
        if np.isnan(act).any() or np.isinf(act).any():
            logger.log("WARNING: nan or inf action {}".format(act))
            return np.zeros_like(act)
        else:
            return act


class DiscreteGenericAgent(GenericAgent):
    def __init__(self, env_dim, act_dim, inner_lr=None, **kwargs):
        assert inner_lr is not None
        super().__init__(env_dim, act_dim, 1, **kwargs)
        self.pi = NN([env_dim] + list([64, 64]) + [act_dim], out_fn=F.softmax)
        self._lst_adam = [Adam(var.shape, stepsize=inner_lr) for var in self.backprop_params]

    @staticmethod
    def cat_sample(prob_matrix):
        s = np.cumsum(prob_matrix, axis=1)[:, :-1]
        r = np.random.rand(prob_matrix.shape[0])
        return (s < r).sum(axis=1)

    @property
    def backprop_params(self):
        return super(DiscreteGenericAgent, self).backprop_params + self.pi.train_vars

    def _pi_f(self, x):
        return [self.pi.f(x)]

    def _pi_logp(self, obs, acts):
        prob = F.sum(self._pi_f(obs)[0] * acts, axis=1)
        return F.log(prob)

    def _logp(self, params, acts):
        prob = F.sum(params[0] * acts, axis=1)
        return F.log(prob)

    def act(self, obs):
        obs = obs.astype(np.float32)
        # Use same normalization as traj.
        traj = np.concatenate(
            [obs, np.zeros(self._act_dim + 2 + len(self.lst_rew_bonus_eval), dtype=np.float32)])

        # Normalize!
        obs = self._traj_norm.norm(traj)[:self._env_dim]
        prob = self.pi.f(obs[np.newaxis, ...]).data
        return int_to_onehot(self.cat_sample(prob)[0], self._act_dim)

    def kl(self, params0, params1):
        return categorical_kl(params0, params1)

    @staticmethod
    def act_to_env_format(act):
        return onehot_to_int(act)
