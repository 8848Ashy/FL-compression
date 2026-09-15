_RANGE_ACC = {}


def _accumulate_range(name, tensor):
    values = tensor.detach()
    _RANGE_ACC.setdefault(name, []).append((values.min().item(), values.max().item(), values.abs().max().item()))


def _print_range_summary():
    for name, values in _RANGE_ACC.items():
        print(name, values[-1])

