# biomol_surface_unsup

Unsupervised neural implicit biomolecular surface learning with VISM-lite objectives.

## Goals
- Input: molecule atoms
- Output: scalar implicit field
- Training: unsupervised / weakly anchored variational objective

## Main commands
- `python scripts/preprocess.py --config configs/data/toy.yaml`
- `python scripts/train.py --config configs/experiment/debug.yaml`
- `python scripts/evaluate.py --ckpt outputs/checkpoints/latest.pt`

