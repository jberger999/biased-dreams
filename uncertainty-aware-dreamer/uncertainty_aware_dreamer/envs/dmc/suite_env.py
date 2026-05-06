import numpy as np
from collections import namedtuple
import os
from typing import Optional


class _Cheetah:

    PROPRIOCEPTIVE_POS_SIZE = 6
    PROPRIOCEPTIVE_VEL_SIZE = 6
    POSITION_SIZE = 8
    DEFAULT_ACTION_REPEAT = 4
    # Separation of position vector into positions and angles
    IS_ANGLE = [False, False, True, True, True, True, True, True, True]
    RANGES = [[None, None], [None, None], [None, None], [np.deg2rad(-30), np.deg2rad(60)], [np.deg2rad(-50), np.deg2rad(50)], [np.deg2rad(-230), np.deg2rad(50)], [np.deg2rad(-57), np.deg2rad(0.4)], [np.deg2rad(-70), np.deg2rad(50)], [np.deg2rad(-28), np.deg2rad(28)]]

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["position"][-6:]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"][-6:]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return state.observation["position"]
    
    @staticmethod
    def get_translation_inv_mask(transformed=True) -> np.ndarray:
        mask = np.ones((25,)) if transformed else np.ones((18,))
        mask[0] = 0.0 # mask torso's x-position rootx
        return mask


class _Walker:

    PROPRIOCEPTIVE_POS_SIZE = 14
    PROPRIOCEPTIVE_VEL_SIZE = 6
    POSITION_SIZE = 15
    DEFAULT_ACTION_REPEAT = 2
    # Separation of position vector into positions and angles
    IS_ANGLE = [False, False, True, True, True, True, True, True, True]
    RANGES = [[None, None], [None, None], [None, None], [np.deg2rad(-20), np.deg2rad(100)], [np.deg2rad(-150), np.deg2rad(0)], [np.deg2rad(-45), np.deg2rad(45)], [np.deg2rad(-20), np.deg2rad(100)], [np.deg2rad(-150), np.deg2rad(0)], [np.deg2rad(-45), np.deg2rad(45)]]

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["orientations"]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"][3:]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return np.concatenate([state.observation["orientations"], np.array([state.observation["height"]])])

    @staticmethod
    def get_translation_inv_mask(transformed=True) -> np.ndarray:
        mask = np.ones((25,)) if transformed else np.ones((18,))
        mask[1] = 0.0 # mask torso's x-position rootx
        return mask

class _BallInCup:

    PROPRIOCEPTIVE_POS_SIZE = 2
    PROPRIOCEPTIVE_VEL_SIZE = 2
    POSITION_SIZE = 4
    DEFAULT_ACTION_REPEAT = 4
    # Separation of position vector into positions and angles
    IS_ANGLE = [False, False]
    RANGES = [[None, None], [None, None]]

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["position"][:2]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"][:2]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return state.observation["position"]
    
    @staticmethod
    def get_translation_inv_mask(transformed=True) -> np.ndarray:
        mask = np.ones((8,)) # do not mask anything since all positions/velocities are "relative" to frame
        return mask


class _Reacher:

    PROPRIOCEPTIVE_POS_SIZE = 2
    PROPRIOCEPTIVE_VEL_SIZE = 2
    POSITION_SIZE = 4
    DEFAULT_ACTION_REPEAT = 4
    # Separation of position vector into positions and angles
    IS_ANGLE = [True, True]
    RANGES = [[None, None], [np.deg2rad(-160), np.deg2rad(160)]]

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["position"]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return np.concatenate([state.observation["position"], state.observation["to_target"]])
    
    @staticmethod
    def get_translation_inv_mask(transformed=True) -> np.ndarray:
        mask = np.ones((6,)) if transformed else np.ones((4,)) # do not mask anything since all positions/velocities are "relative" to frame
        return mask


class _Finger:

    PROPRIOCEPTIVE_POS_SIZE = 2
    PROPRIOCEPTIVE_VEL_SIZE = 2
    POSITION_SIZE = 6
    DEFAULT_ACTION_REPEAT = 2
    # Separation of position vector into positions and angles
    IS_ANGLE = [True, True, False]
    RANGES = [[np.deg2rad(-110), np.deg2rad(110)], [np.deg2rad(-110), np.deg2rad(110)], [None, None]]

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["position"][:2]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"][:2]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return np.concatenate([state.observation["position"], state.observation["touch"]])
    
    @staticmethod
    def get_translation_inv_mask(transformed=True) -> np.ndarray:
        mask = np.ones((8,)) if transformed else np.ones((6,)) # do not mask anything since all positions/velocities are "relative" to frame
        return mask


class _Cartpole:

    PROPRIOCEPTIVE_POS_SIZE = 1
    PROPRIOCEPTIVE_VEL_SIZE = 1
    POSITION_SIZE = 3
    DEFAULT_ACTION_REPEAT = 8
    # Separation of position vector into positions and angles
    IS_ANGLE = [False, True]
    RANGES = [[-1.8, 1.8], [None, None]]

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["position"][:1]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"][:1]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return state.observation["position"]
    
    @staticmethod
    def get_translation_inv_mask(transformed=True) -> np.ndarray:
        mask = np.ones((5,)) if transformed else np.ones((4,)) 
        return mask # do not mask anything since all positions/velocities are "relative" to frame


class _Pendulum:

    PROPRIOCEPTIVE_POS_SIZE = 2
    PROPRIOCEPTIVE_VEL_SIZE = 1
    POSITION_SIZE = 2
    DEFAULT_ACTION_REPEAT = 2
    IS_ANGLE = [True]
    RANGES = [[None, None]]

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["orientation"]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return np.concatenate([state.observation["orientation"], state.observation["velocity"]])

    @staticmethod
    def get_position(state) -> np.ndarray:
        return state.observation["orientation"]
    
    @staticmethod
    def get_translation_inv_mask(transformed=True) -> np.ndarray:
        mask = np.ones((3,)) if transformed else np.ones((2,)) # do not mask anything since all positions/velocities are "relative" to frame
        return mask


class _Hopper:

    PROPRIOCEPTIVE_POS_SIZE = 6
    PROPRIOCEPTIVE_VEL_SIZE = 4
    POSITION_SIZE = 8
    DEFAULT_ACTION_REPEAT = 2
    # Separation of position vector into positions and angles
    IS_ANGLE = [False, False, True, True, True, True, True]
    RANGES = [[None, None], [None, None], [None, None], [np.deg2rad(-30), np.deg2rad(30)], [np.deg2rad(-170), np.deg2rad(10)], [np.deg2rad(5), np.deg2rad(150)], [np.deg2rad(-45), np.deg2rad(45)]]

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return np.concatenate([state.observation["position"][-4:], state.observation["touch"]])

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"][-4:]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return np.concatenate([state.observation["position"], state.observation["touch"]])
    
    @staticmethod
    def get_translation_inv_mask(transformed=True) -> np.ndarray:
        mask = np.ones((19,)) if transformed else np.ones((14,))
        mask[0] = 0.0 # mask torso's x-position rootx
        return mask


class _Quadruped:

    PROPRIOCEPTIVE_POS_SIZE = -1
    PROPRIOCEPTIVE_VEL_SIZE = -1
    POSITION_SIZE = -1
    DEFAULT_ACTION_REPEAT = 2
    # Separation of position vector into positions and angles
    IS_ANGLE = []
    RANGES = []

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def get_position(state) -> np.ndarray:
        raise NotImplementedError
    
    @staticmethod
    def get_translation_inv_mask(transformed=True) -> np.ndarray:
        raise NotImplementedError


class _Manipulator:

    PROPRIOCEPTIVE_POS_SIZE = 16
    PROPRIOCEPTIVE_VEL_SIZE = 8
    POSITION_SIZE = 33
    DEFAULT_ACTION_REPEAT = 1
    # Separation of position vector into positions and angles
    IS_ANGLE = []
    RANGES = []

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["arm_pos"].ravel()

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["arm_vel"]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return np.concatenate([state.observation["arm_pos"].ravel(),     # 16
                               state.observation["touch"],               # 5
                               state.observation["hand_pos"],            # 4
                               state.observation["object_pos"],          # 4
                               state.observation["target_pos"]])         # 4
        
    @staticmethod
    def get_translation_inv_mask(transformed=True) -> np.ndarray:
        mask = np.ones((22,)) # do not mask anything since all positions/velocities are "relative" to frame
        return mask


class SuiteBaseEnv:

    DMCEnvSpec = namedtuple("DMCEnvSpec", "domain_name task_name env_cls")
    DMC_ENV_CLASSES = {"cheetah": _Cheetah,
                       "walker": _Walker,
                       "ball_in_cup": _BallInCup,
                       "reacher": _Reacher,
                       "finger": _Finger,
                       "cartpole": _Cartpole,
                       "pendulum": _Pendulum,
                       "hopper": _Hopper,
                       "quadruped": _Quadruped,
                       "manipulator": _Manipulator}

    def __init__(self,
                 domain_name: str,
                 task_name: str,
                 seed: int):

        os.environ["MUJOCO_GL"] = "egl"
        from dm_control import suite

        self._env = suite.load(domain_name=domain_name,
                               task_name=task_name,
                               task_kwargs={"random": seed})
        self._env_cls = self.DMC_ENV_CLASSES[domain_name]

        self.step = self._env.step
        self.reset = self._env.reset
        self.action_spec = self._env.action_spec
        self.observation_spec = self._env.observation_spec

    def reset(self):
        return self._env.reset()

    def step(self, action: np.ndarray):
        return self._env.step(action)

    @property
    def default_action_repeat(self) -> int:
        return self._env_cls.DEFAULT_ACTION_REPEAT

    def render(self, img_size: tuple[int, int], cam: Optional[str] = None) -> np.ndarray:
        return self._env.physics.render(camera_id=0 if cam is None or cam == "default" else cam,
                                        height=img_size[0],
                                        width=img_size[1])

    @property
    def proprioceptive_pos_size(self):
        return self._env_cls.PROPRIOCEPTIVE_POS_SIZE

    @property
    def proprioceptive_vel_size(self):
        return self._env_cls.PROPRIOCEPTIVE_VEL_SIZE

    @property
    def position_size(self):
        return self._env_cls.POSITION_SIZE
    
    @property
    def phys_state_range(self):
        return self._env_cls.PHYS_DIM_MIN, self._env_cls.PHYS_DIM_MAX
    
    @property
    def ranges(self):
        return self._env_cls.RANGES
    
    @property
    def is_angle(self):
        return self._env_cls.IS_ANGLE

    @staticmethod
    def get_info(state) -> dict:
        return {}

    def get_position(self, state) -> np.ndarray:
        return self._env_cls.get_position(state)

    def get_proprioceptive_position(self, state) -> np.ndarray:
        return self._env_cls.get_proprioceptive_position(state)

    def get_proprioceptive_velocity(self, state) -> np.ndarray:
        return self._env_cls.get_proprioceptive_velocity(state)

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        return self._env.observation_spec()

    @property
    def physics(self):
        return self._env.physics
    
    def get_translation_inv_mask(self, transformed=True) -> np.ndarray:
        return self._env_cls.get_translation_inv_mask(transformed)
