import numpy as np
from gym import Env
from gym.envs.registration import EnvSpec
from gym.wrappers.time_limit import TimeLimit

from epg.envs.mujoco.hopper import RandomWeightHopperEnv, RandomWeightHopperDirEnv, NormalHopperEnv


class RandomHopper(Env):

    def __init__(self,
                 rand_mass=True, rand_gravity=True, rand_friction=True, rand_thickness=True,
                 seed=None, **_):
        self.rand_mass = rand_mass
        self.rand_gravity = rand_gravity
        self.rand_friction = rand_friction
        self.rand_thickness = rand_thickness

        env = RandomWeightHopperEnv(rand_mass=self.rand_mass,
                                    rand_gravity=self.rand_gravity,
                                    rand_friction=self.rand_friction,
                                    rand_thickness=self.rand_thickness)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reset_model = env.reset_model
        self.step = env.step

    def meta_reset(self, seed):
        np.random.seed(seed)

        env = RandomWeightHopperEnv(rand_mass=self.rand_mass,
                                    rand_gravity=self.rand_gravity,
                                    rand_friction=self.rand_friction,
                                    rand_thickness=self.rand_thickness)

        # Based on Hopper-v2
        spec = EnvSpec(
            'RandomWeightHopperEnv-v0',
            entry_point='generic_rl.envs.mujoco:RandomWeightHopperEnv',
            max_episode_steps=1000,
            reward_threshold=3800.0
        )

        env._spec = spec
        env.seed(seed)

        # Wrap the env as needed
        env = TimeLimit(
            env,
            max_episode_steps=spec.max_episode_steps,
            max_episode_seconds=spec.max_episode_seconds
        )

        self.env = env
        # Fix for done flags.
        self.env.reset()
        self.step = env.step
        self.render = env.render
        self.reset = env.reset


class NormalHopper(Env):

    def __init__(self, seed=None, **_):
        env = NormalHopperEnv()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reset_model = env.reset_model
        self.step = env.step

    def meta_reset(self, seed):
        np.random.seed(seed)
        env = NormalHopperEnv()

        # Based on Hopper-v2
        spec = EnvSpec(
            'NormalHopperEnv-v0',
            entry_point='generic_rl.envs.mujoco:NormalHopperEnv',
            max_episode_steps=1000,
            reward_threshold=3800.0
        )

        env._spec = spec
        env.seed(seed)

        # Wrap the env as needed
        env = TimeLimit(
            env,
            max_episode_steps=spec.max_episode_steps,
            max_episode_seconds=spec.max_episode_seconds
        )

        self.env = env
        # Fix for done flags.
        self.env.reset()
        self.step = env.step
        self.render = env.render
        self.reset = env.reset


class DirHopper(Env):

    def __init__(self, seed=None, **__):
        env = RandomWeightHopperDirEnv()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reset_model = env.reset_model
        self.step = env.step

    def meta_reset(self, seed):
        np.random.seed(seed)

        env = RandomWeightHopperDirEnv()

        # Based on Hopper-v2
        spec = EnvSpec(
            'DirHopperEnv-v0',
            entry_point='generic_rl.envs.mujoco:DirHopperEnv',
            max_episode_steps=1000,
            reward_threshold=3800.0
        )

        env._spec = spec
        env.seed(seed)

        # Wrap the env as needed
        env = TimeLimit(
            env,
            max_episode_steps=spec.max_episode_steps,
            max_episode_seconds=spec.max_episode_seconds
        )

        self.env = env
        # Fix for done flags.
        self.env.reset()
        self.step = env.step
        self.render = env.render
        self.reset = env.reset
