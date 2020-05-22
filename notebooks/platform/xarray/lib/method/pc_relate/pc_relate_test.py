import inspect
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd

import pytest
from dask_ml.decomposition import PCA
from hypothesis import given
from hypothesis.extra.numpy import arrays
from lib import api
from pytest import fixture

from .pc_relate import gramian, impute_with_variant_mean, pc_relate


def test_pc_relate_dummy() -> None:
    s = 100
    rnd_pcs = np.random.rand(2, s)
    rnd_g = np.random.rand(1000, s)
    r = pc_relate(rnd_pcs, rnd_g).compute()
    assert np.all(r < 0.5), "Relatedness estimations should be below 0.5"


def test_bad_maf() -> None:
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


@given(arrays(np.int8, (3, 5)))
def test_gramian_is_symmetric(a) -> None:
    b = gramian(a)
    assert np.allclose(b, b.T)


def test_impute_dummy() -> None:
    a = np.asarray([[0, -1, 0], [1, -1, 0], [1, 1, np.nan]])
    ia_mask, ia = impute_with_variant_mean(a)
    assert np.allclose(ia, [[0, 0, 0], [1, 0.5, 0], [1, 1, 1]])
    assert np.allclose(
        ia_mask, [[False, True, False], [False, True, False], [False, False, True]]
    )


@fixture(scope="session")
def plink_data_prefix() -> str:
    this_file = Path(inspect.getfile(test_pc_relate_full_no_missing_values))
    return this_file.parent.joinpath("test_resources").joinpath("data").as_posix()


def test_pc_relate_full_no_missing_values(plink_data_prefix: str) -> None:
    test_resources = Path(plink_data_prefix).parent
    pcs = da.from_array(
        pd.read_csv(
            test_resources.joinpath("pcs.csv").as_posix(), usecols=[1, 2]
        ).to_numpy()
    ).T
    g = api.read_plink(plink_data_prefix).data.data
    phi = pc_relate(pcs, g).compute()
    assert isinstance(phi, np.ndarray)
    assert phi.shape == (1000, 1000)

    # Get genesis/reference results:
    # TODO (rav): this can be way simpler with better test data,
    #             but it might be worth to just wait for data generators
    genesis_phi = pd.read_csv(test_resources.joinpath("kinbtwe.csv"))
    genesis_phi = genesis_phi[["ID1", "ID2", "kin"]]
    genesis_phi["ID1"], genesis_phi["ID2"] = genesis_phi.ID1 - 1, genesis_phi.ID2 - 1
    indices = (genesis_phi["ID1"] * 1000 + genesis_phi["ID2"]).to_numpy()
    values = genesis_phi["kin"].to_numpy()
    genesis_phi_full = np.zeros((1000, 1000))
    np.put(genesis_phi_full, indices, values)

    # Compare with reference/GENESIS:
    genesis_phi_s = genesis_phi_full[np.triu_indices_from(genesis_phi_full, 1)]
    phi_s = phi[np.triu_indices_from(phi, 1)]
    assert len(phi_s) == len(genesis_phi_s)
    assert np.allclose(phi_s, genesis_phi_s)


def test_pc_relate_full_missing_values(plink_data_prefix: str) -> None:
    g = api.read_plink(plink_data_prefix).data.values
    # set 10% to missing
    missing_size = int(g.size * 0.1)
    missing = np.unravel_index(
        np.random.choice(np.arange(g.size), replace=False, size=missing_size), g.shape
    )
    g[missing] = -1
    assert len(g[g < 0.0]) == missing_size
    pcs = PCA(n_components=4).fit(da.from_array(g)).components_[:2]
    phi = pc_relate(pcs, g).compute()
    assert isinstance(phi, np.ndarray)
    assert phi.shape == (1000, 1000)


def test_pc_relate_dup_should_be_0_5(plink_data_prefix: str) -> None:
    g = api.read_plink(plink_data_prefix).data.values
    g[:, -1] = g[:, 0]
    pcs = PCA(n_components=4).fit(da.from_array(g)).components_[:2]
    phi = pc_relate(pcs, g).compute()
    assert isinstance(phi, np.ndarray)
    assert phi.shape == (1000, 1000)
    assert np.allclose(phi[0, -1], 0.5, atol=0.01)
