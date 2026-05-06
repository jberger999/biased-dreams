# Uncertainty-Aware Dreamer

This directory provides the training and evaluation code for the uncertainty-aware Dreamer implementation used in the paper [Biased Dreams: Limitations to Epistemic Uncertainty Quantification in Latent Space Models](https://arxiv.org/abs/2604.25416). It is an adapted version of [this repository](https://github.com/pbecker93/vrkn/tree/main).

## Setup
To get started, simply create the conda environment from the provided `environment.yml` file:
```bash
conda env create -f environment.yml
```

## Running Experiments
We use [hydra](https://hydra.cc/docs/intro/) for configuration management. To run an experiment with the default settings, use
```bash
python main.py algorithm=li-rssm environment=cheetah_run
```
The configuration is mainly separated into two parts:
- `algorithm`: Choice between `li-rssm` (DreamerV1), `li-cat_rssm` (categorical RSSM from DreamerV2), and their uncertainty-aware variants, `li-urssm` and `li-cat_urssm`
- `environment`: Selection of classic DMC environments

The iterate function in `main.py` returns a `log_dict` with all relevant metrics, which can be logged with your preferred logging tool.

## Evaluation
The trained model and agent can be broadly evaluated using
```bash
python eval.py --config_path <path/to/model> --analyze_id True --analyze_ood True --analyze_attr True --analyze_rew True
```
The following analyses are supported:
- `analyze_id`: Find or load most in-distribution (ID) state, analyze model rollouts starting from this state
- `analyze_ood`: Use hard-coded out-of-distribution (OOD) state, analyze model rollouts starting from this state; not available for `cartpole_swingup` environment
- `analyze_attr`: Analyze attractor behavior of both ID and OOD settings; only possible, if analyze_id and `analyze_ood` where executed before
- `analyze_rew`: Analyze reward behavior for random rollouts
- `combined_phys_discr`: Combined analysis of physical discrepancy over all other available runs of same configuration; assumes that `analyze_id` and `analyze_ood` were executed there first
- `combined_rew_discr`: Combined analysis of reward over all other available runs of the same configuration; assumes that `analyze_rew` were executed there first