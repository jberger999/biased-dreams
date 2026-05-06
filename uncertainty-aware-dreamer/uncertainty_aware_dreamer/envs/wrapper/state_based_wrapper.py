from uncertainty_aware_dreamer.envs.wrapper.torch_env_wrapper import TorchEnvWrapper


class StateBasedTorchEnvWrapper(TorchEnvWrapper):
    def __init__(self, *args, **kwargs):
        """
        Add the ability to reset the environment to a specific state.
        """
        super().__init__(*args, **kwargs)
        
    def reset_to_state(self, state, transformed=True):
        np_obs, np_info = self.env.reset_to_state(state, transformed)
        return self._obs_to_torch(np_obs), self._dict_to_torch(np_info)