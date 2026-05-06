import hydra
from omegaconf import OmegaConf
import os
import csv
import torch
import warnings
warnings.filterwarnings('ignore')

from uncertainty_aware_dreamer.experiments.latent_imagination.li_experiment import LIExperiment


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(config):
    if not os.path.isdir(config.log_dir):
        os.makedirs(config.log_dir)
    
    # Dump config file into log directory.
    config_path = config.log_dir + "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)
        
    print("=== General ===")
    print("algorithm:", config.algorithm.name)
    print("seed:", config.seed)
    print("experiment:", config.experiment)
    print("deterministic:", config.deterministic)
    print()

    # Load experiment.
    experiment, infos = LIExperiment(config).create_experiment()
    
    # Train models.
    for i in range(1001):
        log_dict = experiment.iterate(i)
        env_step = i * infos["data_collect_seq"] * 1000 + infos["init_data_collect_seq"] * 1000
        
        # Possibly log log_dict at step env_step using your favorite logging tool.
        
        # Additionally log values locally in a csv file.
        d = {"step": env_step}
        d.update(log_dict)
        if i == 0:
            keys = d.keys()
            with open(config.log_dir + "log.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(keys)
        with open(config.log_dir + "log.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(d.values())
            
        if i % 100 == 0:
            # Intermediate savings.
            if config.save_model:
                experiment.save_models(path=config.log_dir)
            
        if i == 500 and config.half_way:
            warnings.warn("Half-way break-off requested!")
            break
    
    if config.save_dataset:
        dataset = experiment.get_states_actions()
        torch.save(dataset, config.log_dir + "dataset.pt")

    # Save models after training.
    if config.save_model:
        experiment.save_models(path=config.log_dir)
        
    print("Finished training.")


if __name__ == "__main__":
    main()