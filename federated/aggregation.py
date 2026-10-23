import copy

import torch

from utils.communication import normalized_bits_per_dim
from utils.state_dict import flatten_state_dict, unflatten_state_dict, state_dict_add
from federated.local_training import local_train_delta_with_metrics


def federated_round_original_update(global_model, client_deltas):
    flat, shapes = flatten_state_dict(client_deltas[0]); aggregate = flat.clone().zero_()
    for delta in client_deltas: aggregate += flatten_state_dict(delta)[0]
    aggregate /= len(client_deltas)
    global_model.load_state_dict(state_dict_add(global_model.state_dict(), unflatten_state_dict(aggregate, shapes)))
    return 32.0


def federated_round_original(global_model, client_weights_list):
    flat, shapes = flatten_state_dict(client_weights_list[0]); aggregate = flat.clone().zero_()
    for weights in client_weights_list: aggregate += flatten_state_dict(weights)[0]
    aggregate /= len(client_weights_list); global_model.load_state_dict(unflatten_state_dict(aggregate, shapes))
    return 32.0


def run_federated_round(global_model, client_loaders, encoder, *, total_clients=None,
                        clients_per_round=10, round_idx=0, base_seed=42,
                        frame_seed_mode="independent", local_epochs=1,
                        client_lr=0.05, server_lr=1.0):
    """Train selected clients, encode each delta, average decoded updates, apply once."""
    total_clients = len(client_loaders) if total_clients is None else total_clients
    if total_clients != len(client_loaders):
        raise ValueError("total_clients must match client_loaders length")
    if not 1 <= clients_per_round <= total_clients:
        raise ValueError("clients_per_round must be in [1, total_clients]")
    if frame_seed_mode not in ("independent", "shared"):
        raise ValueError("frame_seed_mode must be independent or shared")
    chooser = torch.Generator().manual_seed(int(base_seed) + int(round_idx) * 1000003)
    selected = torch.randperm(total_clients, generator=chooser)[:clients_per_round].tolist()
    global_state = copy.deepcopy(global_model.state_dict())
    aggregate = None
    metrics = []
    update_norms = []
    train_losses = []
    for cid in selected:
        client_seed = int(base_seed) + int(round_idx) * 10007 + cid
        frame_seed = client_seed if frame_seed_mode == "independent" else int(base_seed) + int(round_idx) * 10007
        torch.manual_seed(client_seed)
        delta_state, train_loss = local_train_delta_with_metrics(
            global_model, client_loaders[cid], epochs=local_epochs, lr=client_lr
        )
        flat_delta, shapes = flatten_state_dict(delta_state)
        decoded, info = encoder.encode_decode(flat_delta, seed=frame_seed)
        aggregate = decoded if aggregate is None else aggregate + decoded
        metrics.append(info)
        train_losses.append(train_loss)
        update_norms.append(float(flat_delta.norm().item()))
    aggregate = aggregate / len(selected)
    delta_state = unflatten_state_dict(aggregate, shapes)
    updated = state_dict_add(global_state, delta_state)
    if server_lr != 1.0:
        updated = {key: global_state[key] + (updated[key] - global_state[key]) * server_lr
                   for key in global_state}
    global_model.load_state_dict(updated)
    mean = lambda key: sum(item[key] for item in metrics) / len(metrics)
    return {
        "selected_clients": selected,
        "train_loss": sum(train_losses) / len(train_losses),
        "mean_nmse": mean("nmse"), "mean_mse": mean("mse"),
        "bpd": mean("bpd"), "actual_lam": mean("actual_lam"),
        "client_update_norm": sum(update_norms) / len(update_norms),
    }
