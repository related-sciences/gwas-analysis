from lib import api
import numpy.testing as npt

# Note that shared_datadir is set up by pytest-datadir
def test_load_plink(shared_datadir):
    path = (shared_datadir / "all_chr.maf0.001.N300")
    ds = api.read_plink(path)

    assert "sample" in ds.dims
    assert "variant" in ds.dims
