import torch

from uncertainty_aware_dreamer.envs.wrapper.torch_env_wrapper import TorchEnvWrapper
from uncertainty_aware_dreamer.envs.wrapper.old_gym_interface_wrapper import OldGymInterfaceWrapper
from uncertainty_aware_dreamer.ssm_mbrl.util.config_dict import ConfigDict
from uncertainty_aware_dreamer.envs.dmc.suite_env import SuiteBaseEnv
from uncertainty_aware_dreamer.envs.dmc.dmc_mbrl_env import DMCMBRLEnv, ObsTypes
from uncertainty_aware_dreamer.envs.wrapper.state_based_wrapper import StateBasedTorchEnvWrapper
from uncertainty_aware_dreamer.envs.dmc.state_based_env import StateBasedDMCMBRLEnv


class DMCEnvFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()
        config.env = "cheetah_run"
        config.action_repeat = -1  # env default
        config.obs_type = "img"

        if finalize_adding:
            config.finalize_adding()

        return config

    @staticmethod
    def build(seed: int,
              config: ConfigDict,
              dtype=torch.float32):
        env_list = config.env.split("-")
        env_list = env_list[-1].split("_")
        domain_name = ""
        for s in env_list[:-2]:
            domain_name += s + "_"
        domain_name += env_list[-2]
        task_name = env_list[-1]

        base_env = SuiteBaseEnv(domain_name=domain_name,
                                task_name=task_name,
                                seed=seed)

        obs_types = {"img": ObsTypes.IMAGE,
                     "img_pro_pos": ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION}
        
        env = DMCMBRLEnv(base_env=base_env,
                         seed=seed,
                         action_repeat=config.action_repeat,
                         obs_type=obs_types[config.obs_type],
                         img_size=(64, 64),
                         image_to_info=False)

        env = OldGymInterfaceWrapper(env)
        env = TorchEnvWrapper(env, dtype)
        return env


class StateBasedDMCEnvFactory(DMCEnvFactory):
    
    @staticmethod
    def build(seed: int,
              config: ConfigDict,
              dtype=torch.float32):

        env_list = config.env.split("-")
        env_list = env_list[-1].split("_")
        domain_name = ""
        for s in env_list[:-2]:
            domain_name += s + "_"
        domain_name += env_list[-2]
        task_name = env_list[-1]

        base_env = SuiteBaseEnv(domain_name=domain_name,
                                task_name=task_name,
                                seed=seed)

        obs_types = {"img": ObsTypes.IMAGE,
                     "img_pro_pos": ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION}
        
        # Use the state-based version of the environment here.
        env = StateBasedDMCMBRLEnv(base_env=base_env,
                                   seed=seed,
                                   action_repeat=config.action_repeat,
                                   obs_type=obs_types[config.obs_type],
                                   img_size=(64, 64),
                                   image_to_info=False)

        env = OldGymInterfaceWrapper(env)
        # Use the state-based torch wrapper here.
        env = StateBasedTorchEnvWrapper(env, dtype)
        return env