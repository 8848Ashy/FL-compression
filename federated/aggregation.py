from utils.communication import normalized_bits_per_dim
from utils.state_dict import flatten_state_dict, unflatten_state_dict, state_dict_add


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

