# Federated learning compression experiments

This repository studies communication-efficient updates for federated learning.
The compression implementations live in `compression/`; model, data, and
training code live in `models/`, `data/`, and `federated/`.

## Entry points

- `python experiments/exp1_bit_and_lambda_law.py` runs isolated single-vector
  compression tests across bit widths and redundancy factors. It uses a real
  one-client MNIST update when the local dataset is available. Use
  `--synthetic-only` to skip local training.
- `python experiments/exp2_federated_mnist.py --mode main` runs the existing
  MNIST federated-learning entry point.
- `python experiments/exp2_federated_mnist.py --mode multi-seed` runs the
  multi-seed federated-learning experiment.

Generated artifacts are stored under `outputs/`. Research papers, reports,
slides, and notebooks are stored under `docs/`.

## Tests

Run `python -m pytest` from the project root.
