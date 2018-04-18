import time

import numpy as np

from epg import utils
from epg.launching import logger


def run_batch_rl(env, agent, inner_opt_freq, inner_max_n_epoch, inner_buffer_size, pool_rank, ppo_factor,
                 epoch=None,
                 render=False,
                 verbose=False):
    from collections import deque
    assert isinstance(inner_opt_freq, int)
    assert isinstance(inner_max_n_epoch, int)
    assert isinstance(inner_buffer_size, int)
    lst_ep_rew, lst_loss, lst_ep_steps, lst_kl = [], [], [], []
    buffer = deque(maxlen=inner_buffer_size)
    n_ep, ep_rew, ep_steps = 0, 0., 0
    tot_update_time, start_env_time = 0., time.time()
    # Assumes meta wrapper used.
    if epoch is not None:
        env.meta_reset(epoch)
        env.seed(epoch)
    else:
        env.meta_reset(pool_rank + utils.get_time_seed())
        env.seed(pool_rank + utils.get_time_seed())

    obs = env.reset()
    n_steps = 0
    for itr in range(inner_max_n_epoch):
        ep_obs = []
        for _ in range(inner_opt_freq):
            obs = obs.astype(np.float32)
            act = agent.act(obs)
            obs_prime, rew, done, _ = env.step(agent.act_to_env_format(act))
            ep_obs.append(obs)
            buffer.append((obs, act, rew, done))
            ep_rew += rew
            ep_steps += 1
            n_steps += 1
            if done:
                obs = env.reset()
                lst_ep_rew.append(ep_rew)
                lst_ep_steps.append(ep_steps)
                if verbose and pool_rank == 0:
                    logger.log('Train run (ep {}, return {:.3f})'.format(n_ep, ep_rew))
                ep_steps, ep_rew = 0, 0.
                n_ep += 1
            else:
                obs = obs_prime

        # This is disabled for now. But it's easy to add an exploration bonus as an additional
        # input the the loss function!
        # for rew_bonus_eval in agent.lst_rew_bonus_eval:
        #     rew_bonus_eval.fit_before_process_samples(obs)

        start_update_time = time.time()
        loss_input = [np.array([e[i] for e in buffer], dtype=np.float32) for i in range(len(buffer[0]))]
        loss_input += [ppo_factor, inner_opt_freq]
        loss, kl = agent.update(*loss_input)
        lst_loss.append(loss)
        lst_kl.append(kl)
        tot_update_time += time.time() - start_update_time

    # Evaluate final policy
    obs, final_rew, ep_counter = env.reset(), [0., 0., 0.], 0
    while ep_counter < 3:
        obs = obs.astype(np.float32)
        act = agent.act(obs)
        obs_prime, rew, done, _ = env.step(agent.act_to_env_format(act))
        final_rew[ep_counter] += rew
        if done:
            obs = env.reset()
            ep_counter += 1
        else:
            obs = obs_prime

    tot_env_time = time.time() - start_env_time - tot_update_time

    if render:
        logger.log('Rendering final policy for 5 steps ...')
        obs, ep_rew = env.reset(), 0.
        ep_counter = 0
        while ep_counter < 5:
            obs = obs.astype(np.float32)
            act = agent.act(obs)
            obs_prime, rew, done, _ = env.step(agent.act_to_env_format(act))
            env.render()
            ep_rew += rew
            if done:
                logger.log('Test run with final policy (return {:.3f}).'.format(ep_rew))
                time.sleep(2)
                obs, ep_rew = env.reset(), 0.
                ep_counter += 1
            else:
                obs = obs_prime

    return dict(ep_return=np.asarray(lst_ep_rew),
                ep_final_rew=np.asarray(final_rew),
                ep_loss=lst_loss,
                ep_length=lst_ep_steps,
                ep_kl=np.asarray(lst_kl),
                update_time=tot_update_time,
                env_time=tot_env_time)
