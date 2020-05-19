import numpy as np
import pytest
import dask
import sys
import xarray as xr
from hypothesis import given, strategies as st, settings
from lib.config import config
from lib.stats import core
from lib.stats.core import ld_matrix, axis_intervals
from lib.dispatch import Domain
from lib.core import GenotypeCountDataset
from ..utils import PHASES_NO_SHRINK

ld_matrix_backends = ['dask/numba', 'dask/cuda']
ld_matrix_prop = str(Domain(core.DOMAIN).append(ld_matrix.__name__).append('backend'))

@pytest.fixture(scope="function", params=ld_matrix_backends)
def backend(request):
    # See: https://docs.pytest.org/en/latest/fixture.html
    backend = request.param
    
    # Do not forget to use single-threaded scheduler in unit tests for GPU calcs!
    # You will get 'FakeCUDAModule' object has no attribute 'to_device'
    # when attempting numba.jit from multiple threads
    if 'cuda' in backend.split('/'):
        dask.config.set(scheduler='single-threaded')
    else:
        dask.config.set(scheduler='threads')

    with config.context(ld_matrix_prop, backend):
        yield backend

@pytest.fixture(scope="function", params=[True, False])
def backend_setting(backend, request):
    if backend == 'dask/cuda' and not request.param:
        pytest.skip('Non-preallocated backend not implemented on GPU')
    yield dict(backend=backend, preallocate=request.param)

def ldm_df(x, ais_kwargs={}, ldm_kwargs={}, diag=False, nan=False, **kwargs):
    ds = GenotypeCountDataset.create(x)
    if 'positions' in kwargs and kwargs['positions'] is not None:
        ds = ds.assign(pos=xr.DataArray(kwargs.pop('positions'), dims=['variant']))
    if 'groups' in kwargs and kwargs['groups'] is not None:
        ds = ds.assign(contig=xr.DataArray(kwargs.pop('groups'), dims=['variant']))
    intervals = axis_intervals(ds, **ais_kwargs, backend='numba')
    df = ld_matrix(ds, intervals=intervals, **ldm_kwargs).compute()
    if not diag:
        df = df.pipe(lambda df: df[df['i'] != df['j']])
    if not nan:
        df = df[~df['value'].isnull()]
    return df

@pytest.mark.parametrize("n", [2, 10, 16, 22])
def test_window(backend_setting, n):
    # Create zero row vectors except for 1st and 11th 
    # (make them have non-zero variance)
    x = np.zeros((n, 10), dtype='uint8')
    x[0,:-1] = 1
    x[n//2,:-1] = 1
    # All non-self comparisons are nan except for the above two
    df = ldm_df(
        x, 
        ais_kwargs=dict(window=n, step=n, unit='index'), 
        ldm_kwargs=dict(preallocate=backend_setting['preallocate'])
    )
    assert len(df) == 1
    assert df.iloc[0].tolist() == [0, n//2, 1.0]

    # Do the same with physical distance equal to row index
    positions = np.arange(x.shape[0])
    df = ldm_df(
        x, 
        ais_kwargs=dict(window=n, unit='physical'), 
        ldm_kwargs=dict(preallocate=backend_setting['preallocate']), 
        positions=positions
    )
    assert len(df) == 1
    assert df.iloc[0].tolist() == [0, n//2, 1.0]


def test_backend_setting(backend):
    assert config.get(ld_matrix_prop) == backend

@pytest.mark.parametrize('target_chunk_size', [None, 5])
# Ignore warning related to LD calcs for rows with no variance, e.g.
# "RuntimeWarning: invalid value encountered in greater_equal mask = res >= threshold"
@pytest.mark.filterwarnings('ignore:invalid value encountered in greater_equal')
def test_threshold(backend_setting, target_chunk_size):
    # Create zero row vectors except for 1st and 11th 
    # (make them have non-zero variance)
    x = np.zeros((10, 10), dtype='uint8')
    # Make 3rd and 4th perfectly correlated
    x[2,:-1] = 1
    x[3,:-1] = 1
    # Make 8th and 9th partially correlated with 3/4
    x[7,:-5] = 1
    x[8,:-5] = 1
    df = ldm_df(
        x, 
        ais_kwargs=dict(window=10, step=1, unit='index', target_chunk_size=target_chunk_size), 
        ldm_kwargs=dict(preallocate=backend_setting['preallocate'])
    )
    # Should be 6 comparisons (2->3,7,8 3->7,8 7->8)
    assert len(df) == 6
    # Only 2->3 and 7->8 are perfectly correlated
    assert len(df[df['value'] == 1.0]) == 2
    # Do the same with a threshold
    df = ldm_df(
        x, 
        ais_kwargs=dict(window=10, step=1, unit='index', target_chunk_size=target_chunk_size), 
        ldm_kwargs=dict(preallocate=backend_setting['preallocate'], threshold=.5)
    )
    assert len(df) == 2


@pytest.mark.parametrize("dtype", [
    dtype 
    for k, v in np.sctypes.items() 
    for dtype in v if k in ['int', 'uint']
])
def test_dtypes(backend_setting, dtype):
    # Input matrices should work regardless of integer type
    x = np.zeros((5, 10), dtype=dtype)
    df = ldm_df(
        x, 
        ais_kwargs=dict(window=5), 
        ldm_kwargs=dict(preallocate=backend_setting['preallocate']), 
        diag=True
    )
    assert len(df) == 5


def increasing_ints(draw, n):
    """Draw monotonically increasing list of `n` positive ints"""
    d = st.lists(st.integers(min_value=0, max_value=10), min_size=n, max_size=n)
    return list(np.cumsum(np.array(draw(d))))

@st.composite
def ld_matrix_args(draw):
    n = draw(st.integers(1, 30))
    m = draw(st.integers(1, 30))
    window = draw(st.integers(0, 30))
    step = None if window == 0 else draw(st.integers(min_value=1, max_value=window))
    positions = np.array(increasing_ints(draw, n)) if draw(st.booleans()) else None
    groups = None
    # If using groups, first draw many of them there should be,
    # then create random group assignments for `n` items
    # and make sure the result is sorted
    if draw(st.booleans()):
        ng = draw(st.sampled_from([1, 5, 23]))
        groups = list(range(ng)) * (n // ng + 1)
        groups = np.array(sorted(groups[:n]))
        assert n == len(groups)
    unit = 'index' if positions is None else 'physical'
    return n, m, dict(window=window, step=step, unit=unit), dict(positions=positions, groups=groups)


@given(ld_matrix_args()) # pylint: disable=no-value-for-parameter
@settings(max_examples=50, deadline=None, phases=PHASES_NO_SHRINK)
def test_exhaustive_comparisons(backend, args):
    # Validate that no pair-wise comparisons are skipped
    n, m, ais_kwargs, kwargs = args
    x = np.zeros((n, m), dtype='uint8')
    df = ldm_df(x, ais_kwargs, ldm_kwargs=dict(value_init=-99, preallocate=True), diag=True, **kwargs)
    df_unset = df[df['value'] == -99]
    assert len(df_unset) == 0, \
        f'Found {len(df_unset)} unset values; Examples: {df_unset.head(25)}'
