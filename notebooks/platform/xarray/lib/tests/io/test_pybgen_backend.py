from lib import api
import numpy.testing as npt

# Note that shared_datadir is set up by pytest-datadir
def test_load_bgen(shared_datadir):
    path = (shared_datadir / "example.bgen")
    ds = api.read_bgen(path, backend='pybgen', chunks=(100, 500, 3))

    assert "sample" in ds.dims
    assert "variant" in ds.dims

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
