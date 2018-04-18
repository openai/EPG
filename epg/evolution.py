import matplotlib

matplotlib.use('Agg')
import os
import gym
import time

import numpy as np
from epg.launching import logger
from epg import utils
from epg import plotting
from epg.utils import PiecewiseSchedule
from mpi4py import MPI

from epg.losses import Conv1DLoss
from epg.utils import reseed, get_dims, Adam, relative_ranks
from epg.agents import DiscreteGenericAgent, ContinuousGenericAgent
from epg.rollout import run_batch_rl

gym.logger.set_level(41)

# Statics
NUM_EQUAL_NOISE_VECTORS = 1
NUM_TEST_SAMPLES = 7


class ES(object):
    """Evolution Strategies (ES)
    """

    def __init__(self, env, env_id, inner_opt_freq=None, inner_max_n_epoch=None, inner_opt_batch_size=None,
                 inner_buffer_size=None, inner_n_opt_steps=None, inner_lr=None, inner_use_ppo=None,
                 plot_freq=10, gpi=None, mem=None, **_):
        self._env = env
        self._env_id = env_id

        self._outer_plot_freq = plot_freq
        self._outer_evolve_policy_init = gpi

        self._inner_use_mem = mem
        self._inner_opt_freq = inner_opt_freq
        self._inner_opt_batch_size = inner_opt_batch_size
        self._inner_n_opt_steps = inner_n_opt_steps
        self._inner_max_n_epoch = inner_max_n_epoch
        self._inner_buffer_size = inner_buffer_size
        self._inner_lr = inner_lr
        self._inner_use_ppo = inner_use_ppo
        self._inner_mem_out_size = 32

    def create_agent(self, env, pool_rank):
        # Reseed in multiprocessing env.
        reseed(env, pool_rank)

        env_dim, act_dim, n_output_params = get_dims(self._env)
        if isinstance(env.action_space, gym.spaces.Discrete):
            agent_cls = DiscreteGenericAgent
        else:
            agent_cls = ContinuousGenericAgent
        agent = agent_cls(
            env_dim, act_dim, memory_out_size=self._inner_mem_out_size,
            inner_n_opt_steps=self._inner_n_opt_steps,
            inner_opt_batch_size=self._inner_opt_batch_size, inner_lr=self._inner_lr,
            inner_use_ppo=self._inner_use_ppo, mem=self._inner_use_mem, buffer_size=self._inner_buffer_size)
        return agent

    def init_theta(self, env):
        agent = self.create_agent(env, 0)
        loss = Conv1DLoss(traj_dim_in=agent._traj_in_dim)
        theta = loss.get_params_1d()
        if self._outer_evolve_policy_init:
            theta = np.hstack([theta, agent.pi.get_params_1d()])
        return theta

    @staticmethod
    def save_theta(theta, ext=''):
        save_path = os.path.join(logger.get_dir(), 'thetas')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'theta{}.npy'.format(ext)), theta)

    @staticmethod
    def load_theta(path):
        return np.load(path)

    def test(self, fix_ppo=None, load_theta_path=None, **_):
        def objective(env, theta, pool_rank):
            agent = self.create_agent(env, pool_rank)
            loss_n_params = len(agent.get_loss().get_params_1d())
            agent.get_loss().set_params_1d(theta[:loss_n_params])
            if self._outer_evolve_policy_init:
                agent.pi.set_params_1d(theta[loss_n_params:])
            # Agent lifetime is inner_opt_freq * inner_max_n_epoch
            out = run_batch_rl(env, agent,
                               inner_opt_freq=self._inner_opt_freq,
                               inner_max_n_epoch=self._inner_max_n_epoch,
                               inner_buffer_size=self._inner_buffer_size,
                               pool_rank=0,
                               ppo_factor=1. if fix_ppo else 0.,
                               render=True, verbose=True)

        if load_theta_path is not None:
            try:
                theta = self.load_theta(load_theta_path)
                while True:
                    objective(self._env, theta, 0)
            except Exception as e:
                print(e)
        logger.log('Test run finished.')

    def train(self, outer_n_epoch, outer_l2, outer_std, outer_learning_rate, outer_n_samples_per_ep,
              n_cpu=None, fix_ppo=None, **_):
        # Requires more than 1 MPI process.
        assert MPI.COMM_WORLD.Get_size() > 1
        assert n_cpu is not None
        if fix_ppo:
            ppo_factor_schedule = PiecewiseSchedule([(0, 1.), (int(outer_n_epoch / 16), 0.5)],
                                                    outside_value=0.5)
        else:
            ppo_factor_schedule = PiecewiseSchedule([(0, 1.), (int(outer_n_epoch / 8), 0.)],
                                                    outside_value=0.)

        outer_lr_scheduler = PiecewiseSchedule([(0, outer_learning_rate),
                                                (int(outer_n_epoch / 2), outer_learning_rate * 0.1)],
                                               outside_value=outer_learning_rate * 0.1)

        def objective(env, theta, pool_rank):
            agent = self.create_agent(env, pool_rank)
            loss_n_params = len(agent.get_loss().get_params_1d())
            agent.get_loss().set_params_1d(theta[:loss_n_params])
            if self._outer_evolve_policy_init:
                agent.pi.set_params_1d(theta[loss_n_params:])
            # Agent lifetime is inner_opt_freq * inner_max_n_epoch
            return run_batch_rl(env, agent,
                                inner_opt_freq=self._inner_opt_freq,
                                inner_buffer_size=self._inner_buffer_size,
                                inner_max_n_epoch=self._inner_max_n_epoch,
                                pool_rank=pool_rank,
                                ppo_factor=ppo_factor_schedule.value(epoch),
                                epoch=None)

        # Initialize theta.
        theta = self.init_theta(self._env)
        num_params = len(theta)
        logger.log('Theta dim: {}'.format(num_params))

        # Set up outer loop parameter update schedule.
        adam = Adam(shape=(num_params,), beta1=0., stepsize=outer_learning_rate, dtype=np.float32)

        # Set up intra-machine parallelization.
        logger.log('Using {} proceses per MPI process.'.format(n_cpu))
        from pathos.multiprocessing import ProcessPool
        pool = ProcessPool(nodes=n_cpu)

        begin_time, best_test_return = time.time(), -np.inf
        for epoch in range(outer_n_epoch):

            # Anneal outer learning rate
            adam.stepsize = outer_lr_scheduler.value(epoch)

            noise = np.random.randn(outer_n_samples_per_ep // NUM_EQUAL_NOISE_VECTORS, num_params)
            noise = np.repeat(noise, NUM_EQUAL_NOISE_VECTORS, axis=0)
            theta_noise = theta[np.newaxis, :] + noise * outer_std
            theta_noise = theta_noise.reshape(MPI.COMM_WORLD.Get_size(), -1)

            # Distributes theta_noise vectors to all nodes.
            logger.log('Scattering all perturbed theta vectors and running inner loops ...')

            recvbuf = np.empty(theta_noise.shape[1], dtype='float')
            MPI.COMM_WORLD.Scatter(theta_noise, recvbuf, root=0)
            theta_noise = recvbuf.reshape(-1, num_params)

            # Noise vectors are scattered, run inner loop, parallelized over `pool_size` processes.
            start_time = time.time()
            pool_size = int(outer_n_samples_per_ep / MPI.COMM_WORLD.Get_size())
            results = pool.amap(objective, [self._env] * pool_size, theta_noise, range(pool_size)).get()

            # Extract relevant results
            returns = [utils.ret_to_obj(r['ep_final_rew']) for r in results]
            update_time = [np.mean(r['update_time']) for r in results]
            env_time = [np.mean(r['env_time']) for r in results]
            ep_length = [np.mean(r['ep_length']) for r in results]
            n_ep = [len(r['ep_length']) for r in results]
            mean_ep_kl = [np.mean(r['ep_kl']) for r in results]
            final_rets = [np.mean(r['ep_return'][-3:]) for r in results]

            # We gather the results at node 0
            recvbuf = np.empty([MPI.COMM_WORLD.Get_size(), 7 * pool_size],
                               # 7 = number of scalars in results vector
                               dtype='float') if MPI.COMM_WORLD.Get_rank() == 0 else None
            results_processed_arr = np.asarray(
                [returns, update_time, env_time, ep_length, n_ep, mean_ep_kl, final_rets],
                dtype='float').ravel()
            MPI.COMM_WORLD.Gather(results_processed_arr, recvbuf, root=0)

            # Do outer loop update calculations at node 0
            if MPI.COMM_WORLD.Get_rank() == 0:
                end_time = time.time()
                logger.log(
                    'All inner loops completed, returns gathered ({:.2f} sec).'.format(
                        time.time() - start_time))

                results_processed_arr = recvbuf.reshape(MPI.COMM_WORLD.Get_size(), 7, pool_size)
                results_processed_arr = np.transpose(results_processed_arr, (0, 2, 1)).reshape(-1, 7)
                results_processed = [dict(returns=r[0],
                                          update_time=r[1],
                                          env_time=r[2],
                                          ep_length=r[3],
                                          n_ep=r[4],
                                          mean_ep_kl=r[5],
                                          final_rets=r[6]) for r in results_processed_arr]
                returns = np.asarray([r['returns'] for r in results_processed])

                # ES update
                noise = noise[::NUM_EQUAL_NOISE_VECTORS]
                returns = np.mean(returns.reshape(-1, NUM_EQUAL_NOISE_VECTORS), axis=1)
                theta_grad = relative_ranks(returns).dot(noise) / outer_n_samples_per_ep \
                             - outer_l2 * theta
                theta -= adam.step(theta_grad)

                # Perform `NUM_TEST_SAMPLES` evaluation runs on root 0.
                if epoch % self._outer_plot_freq == 0 or epoch == outer_n_epoch - 1:
                    start_test_time = time.time()
                    logger.log('Performing {} test runs in parallel on node 0 ...'.format(NUM_TEST_SAMPLES))
                    # Evaluation run with current theta
                    test_results = pool.amap(
                        objective,
                        [self._env] * NUM_TEST_SAMPLES,
                        theta[np.newaxis, :] + np.zeros((NUM_TEST_SAMPLES, num_params)),
                        range(NUM_TEST_SAMPLES)
                    ).get()
                    plotting.plot_results(epoch, test_results)
                    test_return = np.mean([utils.ret_to_obj(r['ep_return']) for r in test_results])
                    if test_return > best_test_return:
                        best_test_return = test_return
                        # Save theta as numpy array.
                        self.save_theta(theta)
                    self.save_theta(theta, str(epoch))
                    logger.log('Test runs performed ({:.2f} sec).'.format(time.time() - start_test_time))

                logger.logkv('Epoch', epoch)
                utils.log_misc_stats('Obj', logger, returns)
                logger.logkv('PPOFactor', ppo_factor_schedule.value(epoch))
                logger.logkv('EpochTimeSpent(s)', end_time - start_time)
                logger.logkv('TotalTimeSpent(s)', end_time - begin_time)
                logger.logkv('BestTestObjMean', best_test_return)
                logger.dumpkvs()
