import torch
import numpy as np
from dm_env import TimeStep, StepType

from uncertainty_aware_dreamer.envs.dmc.dmc_mbrl_env import DMCMBRLEnv


class StateBasedDMCMBRLEnv(DMCMBRLEnv):
    def __init__(self, *args, **kwargs):
        """
        Add the ability to reset the environment to a specific state.
        """
        super().__init__(*args, **kwargs)
        
    def get_translation_inv_mask(self, transformed=True):
        return self._base_env.get_translation_inv_mask(transformed)
        
    def get_phys_state(self, mask=False):
        phys = self._base_env.physics
        state = self.transform_phys_state(phys.get_state())
        state = state * self.get_translation_inv_mask() if mask else state
        return state
    
    def get_phys_state_size(self):
        return len(self.get_phys_state())
    
    def get_loss_masks(self, img_shape):
        return [
            np.ones((img_shape[:2])),       # image mask for all 3 dims
            np.ones((1,)),                  # reward mask
            self.get_translation_inv_mask() # phys. state mask
        ]
        
    def get_is_angle(self):
        return self._base_env.is_angle

    def reset(self):
        obs, info = super().reset()
        
        # Add physical state + loss masks to the collected infomation.
        info |= {
            "state": self.get_phys_state(),
            "loss_mask": self.get_loss_masks(img_shape=obs[0].shape)
        }
        
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        
        # Add physical state + loss masks to the collected infomation.
        info |= {
            "state": self.get_phys_state(),
            "loss_mask": self.get_loss_masks(img_shape=obs[0].shape)
        }
        
        return obs, reward, done, truncated, info

    def reset_to_state(self, state, transformed=True):
        # Reset gym environment wrapper.
        _, info = super().reset()
        phys = self._base_env.physics
        
        # If given state is transformed, map it to the original range
        state = self.untransform_phys_state(state) if transformed else state
        
        # Set the physical state.
        with phys.reset_context():
            phys.set_state(state)
            
        # Build observation directly from the task
        raw_obs = self._base_env._env._task.get_observation(phys)

        # Construct a Timestep struct without calling reset
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=None,
            discount=None,
            observation=raw_obs,
        )
        
        # Get updated observation.
        obs = self._get_obs(timestep)
        info = self._get_info(timestep)
        
        # Add physical state + loss masks to the collected infomation.
        info |= {
            "state": self.get_phys_state(),
            "loss_mask": self.get_loss_masks(img_shape=obs[0].shape)
        }
            
        return obs, info
        
    def transform_phys_state(self, state):
        """
        Clip state dimensions, if specified by environment.
        Replace raw angle theta (in radians) with (sin theta, cos theta).
        """
        ranges = self._base_env.ranges
        is_angle = self._base_env.is_angle
        
        new_state = []
        for dim in range(len(ranges) * 2):
            state_dim = state[..., dim]
            if dim < len(ranges):
                # Looking into positions/angles
                if ranges[dim] != [None, None]:
                    # Clamp position/angle dimensions to their respective ranges, if specified
                    state_dim = np.clip(state_dim, ranges[dim][0], ranges[dim][1])
                if is_angle[dim]:
                    # Replace each angle theta with (sin(theta), cos(theta))
                    new_state.append(np.sin(state[..., dim])[..., None])
                    new_state.append(np.cos(state[..., dim])[..., None])
                else:
                    new_state.append(state_dim[..., None])
            else:
                # No changes to velocities
                new_state.append(state_dim[..., None])
        
        return np.concatenate(new_state, -1) if isinstance(state, np.ndarray) else torch.cat(new_state, -1)
            
    def untransform_phys_state(self, state):
        ranges = self._base_env.ranges
        is_angle = self._base_env.is_angle
        
        new_state = []
        for dim in range(len(ranges) * 2):
            # Consider dimension offset introduced by angles during transofrm
            num_angles_before = sum(is_angle[:dim])
            if dim < len(ranges):
                if is_angle[dim]:
                    # Replace each angle (sin(theta), cos(theta)) with theta (in [-pi, pi])
                    angle_state = state[..., dim+num_angles_before:dim+num_angles_before+2]
                    angle_state = torch.atan2(angle_state[..., 0], angle_state[..., 1])
                    new_state.append(angle_state[..., None])
                else:
                    new_state.append(state[..., dim+num_angles_before:dim+num_angles_before+1])
            else:
                new_state.append(state[..., dim+num_angles_before:dim+num_angles_before+1])
        
        return torch.cat(new_state, -1)