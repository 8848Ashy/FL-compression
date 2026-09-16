import math
import torch
from utils.communication import normalized_bits_per_dim, srk_encoded_dimension
from utils.state_dict import flatten_state_dict, unflatten_state_dict, state_dict_add
from compression.quantization import stochastic_k_level_quantize


def _fwht_fast(tensor):
    n = tensor.shape[0]; result = tensor.clone(); h = 1
    while h < n:
        blocks = result.view(-1, 2 * h); left, right = blocks[:, :h].clone(), blocks[:, h:].clone()
        blocks[:, :h], blocks[:, h:] = left + right, left - right; h *= 2
    return result.view(n) / math.sqrt(n)


def federated_round_srk_update(global_model, client_deltas, k_levels, rotation_seed=None,
                               quant_seeds=None, relerr_out=None):
    flat0, shapes = flatten_state_dict(client_deltas[0]); d = flat0.shape[0]; D = srk_encoded_dimension(d)
    device = flat0.device
    generator = torch.Generator(); generator.manual_seed(rotation_seed if rotation_seed is not None else torch.seed())
    signs = ((torch.rand(D, generator=generator) < .5).float() * 2 - 1).to(device); aggregate = torch.zeros(D, device=device)
    for index, delta in enumerate(client_deltas):
        flat, _ = flatten_state_dict(delta); padded = torch.zeros(D, device=device); padded[:d] = flat
        rotated = _fwht_fast(padded * signs)
        quant_generator = None
        if quant_seeds is not None:
            quant_generator = torch.Generator(device=device); quant_generator.manual_seed(quant_seeds[index])
        quantized, _ = stochastic_k_level_quantize(rotated, k_levels, generator=quant_generator); aggregate += quantized
        if relerr_out is not None:
            reconstructed = _fwht_fast(quantized) * signs
            relerr_out.append(float(((reconstructed[:d] - flat) ** 2).sum() / (flat ** 2).sum().clamp_min(1e-12)))
    aggregate /= len(client_deltas); recovered = _fwht_fast(aggregate) * signs
    global_model.load_state_dict(state_dict_add(global_model.state_dict(), unflatten_state_dict(recovered[:d], shapes)))
    return normalized_bits_per_dim(D, d, int(math.ceil(math.log2(k_levels))))

