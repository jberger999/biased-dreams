import numpy as np
import torch


OOD_STATES = {
    "cheetah_run": 
        torch.tensor([100, -0.2, np.deg2rad(135), 0, 0, 0, 0, 0, 0] + [5, 0, 10, 0, 0, 0, 0, 0, 0]),
    "hopper_hop":
        torch.tensor([100, -0.6, np.deg2rad(135), 0, 0, 0, 0] + [5, 0, 5, 0, 0, 0, 0]),
    "walker_run":
        torch.tensor([-1.2, 100, np.deg2rad(-90), 0, 0, 0, 0, 0, 0] + [0, 0, -5, 0, 0, 0, 0, 0, 0]),
    "cartpole_swingup":
        # State not defined, since we assume sufficiently explored state space.
        None
}

def get_hardcoded_ood_state(env_name: str):
    assert env_name in ["cheetah_run", "hopper_hop", "walker_run", "cartpole_swingup"], "[ERROR] OOD state for " + env_name + " not defined."
    return OOD_STATES.get(env_name)