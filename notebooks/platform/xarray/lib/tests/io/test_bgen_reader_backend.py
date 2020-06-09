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

    # check some of the data (in different chunks)
    npt.assert_almost_equal(ds.data.values[1][0][0], 0.005, decimal=3)
    npt.assert_almost_equal(ds.data.values[100][0][0], 0.993, decimal=3)
