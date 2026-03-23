import argparse
import yaml

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_experiment_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    exp = load_yaml(args.config)
    return {
        "experiment": exp,
        "data": load_yaml(exp["data"]["config"]),
        "model": load_yaml(exp["model"]["config"]),
        "loss": load_yaml(exp["loss"]["config"]),
        "train": load_yaml(exp["train"]["config"]),
    }

def load_eval_config():
    return load_experiment_config()