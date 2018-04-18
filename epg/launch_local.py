import datetime
import os

import click
import numpy as np
from mpi4py import MPI

from epg.launching import launcher, logger
from epg.envs.random_robots import RandomHopper, DirHopper, NormalHopper
from epg.evolution import ES

"""
Evolved Policy Gradients (EPG)
------------------------------

Run via:
cd <path_to_EPG_folder/EPG>
PYTHONPATH=. python epg/launch_local.py

Test via:
PYTHONPATH=. python epg/launch_local.py --test true
"""


def env_selector(env_id, seed=0):
    if 'RandomHopper' == env_id:
        env = RandomHopper(seed=seed)
    elif 'DirHopper' == env_id:
        env = DirHopper(seed=seed)
    elif 'NormalHopper' == env_id:
        env = NormalHopper(seed=seed)
    else:
        raise Exception('Unknown environment.')
    return env


def setup_es(seed=0, env_id='DirHopper', log_path='/tmp/out', n_cpu=1, **agent_args):
    seed = MPI.COMM_WORLD.Get_rank() * 1000
    assert agent_args is not None
    np.random.seed(seed)
    env = env_selector(env_id, seed)
    env.seed(seed)
    es = ES(env, env_id, **agent_args)
    logger.log('Experiment configuration: {}'.format(str(locals())))
    return es


def test_run(seed=0, env_id='DirHopper', log_path='/tmp/out', n_cpu=1, **agent_args):
    es = setup_es(seed, env_id, log_path, n_cpu, **agent_args)
    es.test(**agent_args, n_cpu=n_cpu)


def run(seed=0, env_id='DirHopper', log_path='/tmp/out', n_cpu=1, **agent_args):
    es = setup_es(seed, env_id, log_path, n_cpu, **agent_args)
    es.train(**agent_args, n_cpu=n_cpu)


@click.command()
@click.option("--test", type=bool, default=False)
def main(test):
    d = datetime.datetime.now()
    date = '{}-{}'.format(d.month, d.day)
    time = '{:02d}-{:02d}'.format(d.hour, d.minute)

    # Experiment params
    # -----------------
    env_id = 'DirHopper'
    # Number of noise vector seeds for ES
    outer_n_samples_per_ep = 8
    # Perform policy SGD updates every `inner_opt_freq` steps
    inner_opt_freq = 64
    # Perform `inner_max_n_epoch` total SGD policy updates,
    # so in total `inner_steps` = `inner_opt_freq` * `inner_max_n_epoch`
    inner_max_n_epoch = 128
    # Temporal convolutions slide over buffer of length `inner_buffer_size`
    inner_buffer_size = inner_opt_freq * 8
    # Use PPO bootstrapping?
    ppo = True
    # Evolve policy initialization togeher with loss function?
    gpi = False
    # Fix PPO alpha (ppo_factor) to 0.5?
    fix_ppo = False
    # Use memory structure?
    mem = False
    # Number of outer loop epochs
    outer_n_epoch = 2000
    # Outer loop theta L2 penalty
    outer_l2 = 0.001
    # Outer loop noise standard deviation
    outer_std = 0.01
    # Outer loop Adam step size
    outer_learning_rate = 1e-2
    # Inner loop batch size per gradient update
    inner_opt_batch_size = 32
    # Number of times to cycle through the sampled dataset in the inner loop
    inner_n_opt_steps = 1
    # Inner loop adam step size
    inner_lr = 1e-3
    # Plotting frequency in number of outer loop epochs
    plot_freq = 50
    # Maximum number of cpus used per MPI process
    max_cpu = 2
    # Local experiment log path
    launcher.LOCAL_LOG_PATH = os.path.expanduser("~/EPG_experiments")
    # Where to load theta from for `--test true` purposes
    theta_load_path = '~/EPG_experiments/<path_to_theta.npy>/theta.npy'
    # -----------------

    exp_tag = '{}-{}-{}{}{}{}'.format(
        outer_n_samples_per_ep,
        inner_opt_freq,
        inner_max_n_epoch,
        '-p' if ppo else '',
        '-i' if gpi else '',
        '-f' if fix_ppo else '',
    ).replace('.', '')
    exp_name = '{}-{}-{}'.format(time, env_id.lower(), exp_tag)
    job_name = 'epg-{}--{}'.format(date, exp_name)

    epg_args = dict(
        env_id=env_id,
        n_cpu=max_cpu,
        log_path=os.path.join(launcher.LOCAL_LOG_PATH, date, exp_name),
        load_theta_path=theta_load_path if test else None,
        plot_freq=plot_freq,
        outer_n_epoch=outer_n_epoch,
        outer_l2=outer_l2,
        outer_std=outer_std,
        outer_learning_rate=outer_learning_rate,
        outer_n_samples_per_ep=outer_n_samples_per_ep,
        inner_opt_freq=inner_opt_freq,
        inner_max_n_epoch=inner_max_n_epoch,
        inner_opt_batch_size=inner_opt_batch_size,
        inner_buffer_size=inner_buffer_size,
        inner_n_opt_steps=inner_n_opt_steps,
        inner_lr=inner_lr,
        mem=mem,
        inner_use_ppo=ppo,
        fix_ppo=fix_ppo,
        gpi=gpi,
    )

    mpi_machines = 1
    mpi_proc_per_machine = int(np.ceil(outer_n_samples_per_ep / mpi_machines / float(max_cpu)))
    logger.log(
        'Running experiment {}/{} with {} noise vectors on {} machines with {}'
        ' MPI processes per machine, each using {} pool processes.'.format(
            date, exp_name, outer_n_samples_per_ep, mpi_machines, mpi_proc_per_machine, max_cpu))

    # Experiment launcher
    launcher.call(job_name=job_name,
                  fn=test_run if test else run,
                  kwargs=epg_args,
                  log_relpath=os.path.join(date, exp_name),
                  mpi_proc_per_machine=mpi_proc_per_machine,
                  mpi_machines=mpi_machines)


if __name__ == '__main__':
    main()
