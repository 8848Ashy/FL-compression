from compression.kashin_frame import FourierKashinFrame
from compression.kashin_solver import kashin_solve
from compression.quantization import uniform_quantize


def run_lambda_search(lambda_values, bits_list, input_delta, seed=2026, iterations=10):
    d = input_delta.numel(); results = []
    for value in lambda_values:
        frame = FourierKashinFrame(d, round(value * d), seed=seed); coefficients = kashin_solve(frame, input_delta, iterations)
        for bits in bits_list:
            reconstructed = frame.frame_synthesis(uniform_quantize(coefficients, bits))
            results.append({"lambda": value, "bits": bits, "D": frame.D, "mse": ((reconstructed - input_delta) ** 2).mean().item()})
    return results

