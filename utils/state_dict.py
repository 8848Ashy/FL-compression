import copy
import numpy as np
import torch


def flatten_state_dict(state_dict):
    tensors, shapes = [], {}
    for key, value in state_dict.items():
        tensors.append(value.flatten()); shapes[key] = value.shape
    return torch.cat(tensors), shapes


def unflatten_state_dict(flat_tensor, shapes):
    result, index = {}, 0
    for key, shape in shapes.items():
        size = int(np.prod(shape)); result[key] = flat_tensor[index:index + size].view(shape); index += size
    return result


def state_dict_subtract(local_state, global_state):
    return {key: local_state[key] - global_state[key] for key in global_state}


def state_dict_add(global_state, delta_state):
    return {key: global_state[key] + delta_state[key] for key in global_state}

