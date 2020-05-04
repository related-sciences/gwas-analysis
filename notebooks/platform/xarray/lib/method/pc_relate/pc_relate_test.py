from .pc_relate import pc_relate
import numpy as np


def test_pc_relate_dummy():
    s = 100
    rnd_pcs = np.random.rand(2, s)
    rnd_g = np.random.rand(1000, s)
    r = pc_relate(rnd_pcs, rnd_g).compute()
    assert np.all(r < 0.5), "Relatedness estimations should be below 0.5"
