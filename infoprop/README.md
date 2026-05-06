# Infoprop

This directory provides the training and evaluation code for _Infoprop_ used in the paper [Biased Dreams: Limitations to Epistemic Uncertainty Quantification in Latent Space Models](https://arxiv.org/abs/2604.25416). It is an adapted version of the [original repository](https://github.com/berndfrauenknecht/infoprop/tree/master).

## Setup
To get started, create the conda environment from the provided environment.yml file, activate it, and install the `mbrl` package:
```bash
conda env create -f environment.yml
conda activate infoprop
pip install -e .
```

## Running Experiments
We use [hydra](https://hydra.cc/docs/intro/) for configuration management and [Weights & Biases](https://wandb.ai/site) for experiment tracking and logging. To run an experiment with the default settings, use
```bash
python -m mbrl.examples.main
```

## Evaluation
After training, collect model rollouts from the trained model:
```bash
python rollout_model.py --load_path <path/to/exp>
```
The collected rollouts can then be evaluated and plotted with
```bash
python eval.py --env <env_name> --load_path <path/to/exp> --save_path <path/to/eval>
```