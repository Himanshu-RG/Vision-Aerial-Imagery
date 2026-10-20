import random

import numpy as np
import pytest

from aerial_ovd.utils.seed import set_seed


def _draw():
    out = [random.random(), float(np.random.rand())]
    try:
        import torch

        out.append(float(torch.rand(1)))
    except ImportError:
        pass
    return out


def test_same_seed_same_numbers():
    set_seed(123)
    a = _draw()
    set_seed(123)
    b = _draw()
    assert a == b


def test_different_seed_different_numbers():
    set_seed(1)
    a = _draw()
    set_seed(2)
    b = _draw()
    assert a != b


def test_record_and_validation():
    rec = set_seed(7, deterministic=True)
    assert rec["seed"] == 7 and rec["deterministic"] is True
    with pytest.raises(ValueError):
        set_seed(-1)
    with pytest.raises(ValueError):
        set_seed(1.5)  # type: ignore[arg-type]
