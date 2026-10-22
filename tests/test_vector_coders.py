import pytest
import torch

from compression.vector_coders import (
    METHODS,
    evaluate_single_vector_compression,
    prepare_single_vector_compression,
    evaluate_prepared_vector_compression,
)


@pytest.mark.parametrize("method", METHODS)
def test_single_vector_coders_report_finite_metrics(method):
    x = torch.randn(127, generator=torch.Generator().manual_seed(8))
    result = evaluate_single_vector_compression(
        x, method, lam=1.5, bits=3, quant_seed=19, frame_seed=23
    )
    assert result["actual_bits"] == result["D"] * 3 + 64
    assert result["D"] >= x.numel()
    assert result["kappa_inf"] > 0
    for key in ("alg_nmse", "quant_nmse", "cross_term", "total_nmse"):
        assert torch.isfinite(torch.tensor(result[key]))
    assert result["total_nmse"] == pytest.approx(
        result["alg_nmse"] + result["quant_nmse"] + result["cross_term"],
        rel=1e-5, abs=1e-7,
    )


def test_quant_seed_changes_rounding_but_frame_is_reusable():
    x = torch.randn(96, generator=torch.Generator().manual_seed(3))
    prepared = prepare_single_vector_compression(
        x, "fourier_srk", lam=1.5, frame_seed=41
    )
    first = evaluate_prepared_vector_compression(x, prepared, 1, 3)
    second = evaluate_prepared_vector_compression(x, prepared, 1, 7)
    assert first["D"] == second["D"]
    assert first["alg_nmse"] == second["alg_nmse"]
    assert first["quant_nmse"] != second["quant_nmse"]


@pytest.mark.parametrize("x,method,bits", [
    (torch.zeros(8), "hadamard_srk", 1),
    (torch.ones(8), "unknown", 2),
])
def test_single_vector_coder_rejects_invalid_input(x, method, bits):
    with pytest.raises(ValueError):
        evaluate_single_vector_compression(x, method, 1.0, bits, 1)

