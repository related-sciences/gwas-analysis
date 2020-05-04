import pytest

from .pc_relate import pc_relate
import numpy as np


def test_pc_relate_dummy():
    s = 100
    rnd_pcs = np.random.rand(2, s)
    rnd_g = np.random.rand(1000, s)
    r = pc_relate(rnd_pcs, rnd_g).compute()
    assert np.all(r < 0.5), "Relatedness estimations should be below 0.5"


def test_bad_maf():
    pcs = np.random.rand(2, 10)
    g = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        pc_relate(pcs, g, 1.0)
    with pytest.raises(ValueError):
        pc_relate(pcs, g, 0.0)
    with pytest.raises(ValueError):
        pc_relate(pcs, g, -1.0)
    with pytest.raises(ValueError):
        pc_relate(pcs, g, 2.0)
