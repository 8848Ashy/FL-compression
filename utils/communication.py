import math


def quantized_payload_bits(encoded_dim, bits, metadata_bits=64.0):
    return encoded_dim * bits + metadata_bits


def normalized_bits_per_dim(encoded_dim, original_dim, bits, metadata_bits=64.0):
    return quantized_payload_bits(encoded_dim, bits, metadata_bits) / original_dim


def total_communication_bits(per_client_per_round_bits, num_clients, rounds):
    return per_client_per_round_bits * num_clients * rounds


def total_communication_mb(total_bits):
    return total_bits / 8 / 1024 / 1024


def srk_encoded_dimension(original_dim):
    return 2 ** math.ceil(math.log2(original_dim))


def kashin_encoded_dimension(original_dim, lambda_value):
    return round(lambda_value * original_dim)

