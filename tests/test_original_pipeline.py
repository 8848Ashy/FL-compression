import torch
from models.mnist_mlp import MNIST_MLP
from federated.aggregation import federated_round_original_update


def test_original_update_runs():
    model = MNIST_MLP(); state = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    assert federated_round_original_update(model, [state, state]) == 32.0

