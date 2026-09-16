"""Checks for CRN pairing hooks and per-client compression relerr instrumentation."""
import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.mnist_mlp import MNIST_MLP
from compression.quantization import stochastic_k_level_quantize
from compression.srk import federated_round_srk_update
from compression.kashin_frame import FourierKashinFrame
from compression.kashin_compressor import federated_round_kashin_update


def _random_deltas(model, n=3, scale=0.01):
    return [{k: scale * torch.randn_like(v) for k, v in model.state_dict().items()} for _ in range(n)]


def main():
    loader = DataLoader(TensorDataset(torch.arange(12)), batch_size=4, shuffle=True, generator=torch.Generator())
    loader.generator.manual_seed(555); order1 = [batch[0].tolist() for batch in loader]
    loader.generator.manual_seed(555); order2 = [batch[0].tolist() for batch in loader]
    assert order1 == order2
    print("[PASS] DataLoader shuffle order reproducible under re-seeded generator (CRN pairing hook)")

    torch.manual_seed(7)
    x = torch.randn(4096)
    gen_a = torch.Generator(); gen_a.manual_seed(123)
    gen_b = torch.Generator(); gen_b.manual_seed(123)
    quant_a, _ = stochastic_k_level_quantize(x, 4, generator=gen_a)
    quant_b, _ = stochastic_k_level_quantize(x, 4, generator=gen_b)
    assert torch.equal(quant_a, quant_b)
    print("[PASS] stochastic quantizer reproducible under fixed generator")

    model = MNIST_MLP(); deltas = _random_deltas(model)
    out_a, out_b = [], []
    model_a, model_b = copy.deepcopy(model), copy.deepcopy(model)
    federated_round_srk_update(model_a, deltas, 4, rotation_seed=99, quant_seeds=[11, 12, 13], relerr_out=out_a)
    federated_round_srk_update(model_b, deltas, 4, rotation_seed=99, quant_seeds=[11, 12, 13], relerr_out=out_b)
    assert all(torch.equal(model_a.state_dict()[k], model_b.state_dict()[k]) for k in model_a.state_dict())
    assert out_a == out_b and len(out_a) == 3 and all(0.0 < e < 5.0 for e in out_a)
    print(f"[PASS] SRK quant_seeds deterministic, relerr2={['%.4f' % e for e in out_a]}")

    d = sum(p.numel() for p in model.parameters())
    frame = FourierKashinFrame(d, 101780, seed=2026)
    out_k = []
    federated_round_kashin_update(copy.deepcopy(model), deltas, 4, frame, iterations=10, quant_seeds=[11, 12, 13], relerr_out=out_k)
    assert len(out_k) == 3 and all(0.0 < e < 5.0 for e in out_k)
    print(f"[PASS] Kashin relerr2 collected, SRK mean={sum(out_a) / 3:.4f} vs Kashin mean={sum(out_k) / 3:.4f}")

    default_err = []
    federated_round_srk_update(copy.deepcopy(model), deltas, 4, rotation_seed=99, relerr_out=default_err)
    assert len(default_err) == 3
    print("[PASS] relerr_out works without quant_seeds (backward compatible)")
    print("[PASS] CRN/relerr validation complete")


if __name__ == "__main__":
    main()
