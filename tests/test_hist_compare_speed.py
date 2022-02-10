"""
    This test reproduced the bug where creating a histogram from pytorch tensors
    Takes very (!) long.
    With matplotlib 3.0.2 works fine but with 3.3.4 SUPER slow.
"""


from gputils.startup_guyga import hist_compare, np
import torch
import pytest


def test_hist_compare_speed():
    with pytest.raises(Exception):
        hist_compare({1: list(torch.randn(25000))}, 100)


# def test_hist_compare_speed_list(benchmark):
#     benchmark(hist_compare, {1: list(torch.randn(25000))}, 100)


# def test_hist_compare_speed(benchmark):
#     benchmark(hist_compare, {1: torch.randn(25000)}, 100)