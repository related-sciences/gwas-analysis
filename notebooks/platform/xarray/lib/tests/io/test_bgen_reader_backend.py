import dask
from lib import api
import numpy.testing as npt
import xarray as xr

def test_load_bgen(shared_datadir):
    path = (shared_datadir / "example.bgen")
    ds = api.read_bgen(path, backend='bgen-reader', chunks=(100, 500, 3))

    assert "sample" in ds.dims

    assert "data" in ds.variables
    assert "sample_id" in ds.variables
    assert "variant_id" in ds.variables
    assert "contig" in ds.variables
    assert "pos" in ds.variables
    assert "a1" in ds.variables
    assert "a2" in ds.variables

    # check some of the data (in different chunks)
    npt.assert_almost_equal(ds.data.values[1][0][0], 0.005, decimal=3)
    npt.assert_almost_equal(ds.data.values[100][0][0], 0.916, decimal=3)


def test_load_bgen_compare_backends(shared_datadir):
    path = (shared_datadir / "example.bgen")

    ds_br = api.read_bgen(path, backend='bgen-reader', chunks=(100, 500, 3))
 
    ds_pb = api.read_bgen(path, backend='pybgen', chunks=(100, 500, 3))

    npt.assert_almost_equal(ds_br["data"].values, ds_pb["data"].values)


def test_load_bgen_with_sample_file(shared_datadir):
    path = (shared_datadir / "complex.bgen")
    ds = api.read_bgen(path, backend='bgen-reader')
    # Check the sample IDs are the ones from the .sample file
    assert ds["sample_id"].values.tolist() == ["s0", "s1", "s2", "s3"]
