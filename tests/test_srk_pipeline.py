import torch
from models.mnist_mlp import MNIST_MLP
from compression.srk import federated_round_srk_update


def test_srk_update_runs():
    model = MNIST_MLP(); delta = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    bits = federated_round_srk_update(model, [delta, delta], 4, rotation_seed=2026)
    assert bits > 2.0

