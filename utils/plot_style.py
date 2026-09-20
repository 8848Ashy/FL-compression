"""Shared display settings for experiment figures."""

import matplotlib as mpl


def configure_chinese_plotting():
    """Use an available CJK font while retaining technical labels verbatim."""
    mpl.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "DejaVu Sans",
    ]
    mpl.rcParams["axes.unicode_minus"] = False
