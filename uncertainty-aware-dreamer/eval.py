from hydra import initialize, compose
import argparse
import warnings
warnings.filterwarnings("ignore")

from evaluation.strategy.broad_analysis import evaluate


def main(args):
    # Load the configuration file.
    with initialize(config_path=args.config_path, version_base=None):
        config = compose(config_name=args.config_name)

    print("=== General ===")
    print("algorithm:", config.algorithm.name)
    print("seed:", config.seed)
    print("experiment:", config.experiment)
    print("deterministic:", config.deterministic)
    print()

    # Create inital experiment.
    if "li" in config.algorithm.name:
        from uncertainty_aware_dreamer.experiments.latent_imagination.li_experiment import LIExperiment
        experiment, _ = LIExperiment(config).create_experiment(verbose=False)
    else:
        raise NotImplementedError(f"Algorithm {config.algorithm.name} not implemented.")
    
    # Load trained models.
    experiment.load_models(path=config.log_dir)

    # Evaluate.
    evaluate(experiment=experiment,
             overrides_config=config,
             warm_up_steps=args.warm_up_steps,
             num_init_episodes=args.num_init_episodes,
             rollout_length=args.rollout_length,
             num_rollouts=args.num_rollouts,
             start_state_path=args.start_state_path,
             analyze_id=args.analyze_id,
             analyze_ood=args.analyze_ood,
             analyze_attr=args.analyze_attr,
             analyze_rew=args.analyze_rew,
             combined_phys_discr=args.combined_phys_discr,
             combined_rew_discr=args.combined_rew_discr,
             open_loop=False)
    
    print("Evaluation done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_path", default="out/test-run/li-rssm/cheetah_run/0")
    parser.add_argument("--config_name", default="config")
    
    parser.add_argument("--warm_up_steps", type=int, default=3)
    parser.add_argument("--num_init_episodes", type=int, default=1)
    parser.add_argument("--rollout_length", type=int, default=50)
    parser.add_argument("--num_rollouts", type=int, default=100)
    parser.add_argument("--start_state_path", type=str, default=None)
    
    parser.add_argument("--analyze_id", type=bool, default=True)
    parser.add_argument("--analyze_ood", type=bool, default=True)
    parser.add_argument("--analyze_attr", type=bool, default=True)
    parser.add_argument("--analyze_rew", type=bool, default=True)
    
    parser.add_argument("--combined_phys_discr", type=bool, default=False, help="Combined analysis of physical discrepancy over all other available seeded runs.")
    parser.add_argument("--combined_rew_discr", type=bool, default=False, help="Combined analysis of reward over all other available seeded runs.")
    
    args = parser.parse_args()

    main(args)