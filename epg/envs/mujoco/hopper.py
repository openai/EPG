import os

import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv, HopperEnv


class NormalHopperEnv(HopperEnv):
    def __init__(self, xml_filename="hopper.xml"):
        utils.EzPickle.__init__(self)
        assets_path = os.path.join(os.path.dirname(__file__), "assets")
        xml_path = os.path.join(assets_path, xml_filename)

        MujocoEnv.__init__(self, xml_path, 2)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        reward = (posafter - posbefore) / self.dt
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .5))
        ob = self._get_obs()
        return ob, reward, done, {}


class RandomWeightHopperEnv(HopperEnv):
    def __init__(self, xml_filename="hopper.xml"):
        utils.EzPickle.__init__(self)
        assets_path = os.path.join(os.path.dirname(__file__), "assets")
        xml_path = os.path.join(assets_path, xml_filename)

        self.direction = [-1, 1][np.random.randint(1, 2)]
        MujocoEnv.__init__(self, xml_path, 2)

        self.body_mass_length = len(self.model.body_mass)
        self.geom_friction_length = len(self.model.geom_friction)
        self.geom_size_length = len(self.model.geom_size)

        # Example environment randomizations
        self.random_mass()
        self.random_gravity()
        self.random_friction()
        self.random_thickness()

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        reward = self.direction * (posafter - posbefore) / self.dt
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .5))
        ob = self._get_obs()
        return ob, reward, done, {}

    def random_mass(self):
        for i in range(1, self.body_mass_length):
            self.model.body_mass[i] = self.np_random.uniform(0.5, 2) * self.model.body_mass[i]

    def random_gravity(self):
        self.model.opt.gravity[2] = -self.np_random.uniform(0, 18) - 2

    def random_friction(self):
        for i in range(1, self.geom_friction_length):
            self.model.geom_friction[i, 0] = self.np_random.uniform(0.5, 2) * self.model.geom_friction[i, 0]
            self.model.geom_friction[i, 1] = self.np_random.uniform(0.5, 2) * self.model.geom_friction[i, 1]
            self.model.geom_friction[i, 2] = self.np_random.uniform(0.5, 2) * self.model.geom_friction[i, 2]

    def random_thickness(self):
        for i in range(1, self.geom_size_length):
            self.model.geom_size[i, 0] = self.np_random.uniform(0.5, 2) * self.model.geom_size[i, 0]


class RandomWeightHopperDirEnv(HopperEnv):
    def __init__(self, xml_filename="hopper.xml"):
        utils.EzPickle.__init__(self)
        assets_path = os.path.join(os.path.dirname(__file__), "assets")
        xml_path = os.path.join(assets_path, xml_filename)

        self.direction = [-1, 1][np.random.randint(0, 1)]
        MujocoEnv.__init__(self, xml_path, 2)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        reward = self.direction * (posafter - posbefore) / self.dt
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7))
        ob = self._get_obs()
        return ob, reward, done, {}
