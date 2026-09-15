import math
from utils.communication import normalized_bits_per_dim
from utils.state_dict import flatten_state_dict, unflatten_state_dict, state_dict_add
from compression.quantization import stochastic_k_level_quantize
from compression.kashin_solver import kashin_solve


def federated_round_kashin_update(global_model, client_deltas, k_levels, frame, iterations=10):
    flat0, shapes = flatten_state_dict(client_deltas[0]); aggregate = None
    for delta in client_deltas:
        flat, _ = flatten_state_dict(delta); coefficients = kashin_solve(frame, flat, iterations=iterations)
        quantized, _ = stochastic_k_level_quantize(coefficients, k_levels)
        aggregate = quantized if aggregate is None else aggregate + quantized
    aggregate /= len(client_deltas); delta = unflatten_state_dict(frame.frame_synthesis(aggregate), shapes)
    global_model.load_state_dict(state_dict_add(global_model.state_dict(), delta))
    return normalized_bits_per_dim(frame.D, flat0.shape[0], int(math.ceil(math.log2(k_levels))))

