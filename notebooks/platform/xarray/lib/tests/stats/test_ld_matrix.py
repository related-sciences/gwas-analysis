import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from lib.stats.ld_matrix.dask_backend import ld_matrix

# TODO: Use pytest.metafunc and fixtures to parameterize these same functions across backends

 
def ldm_df(*args, diag=False, nan=False, **kwargs):
    # Do not forget to use single-threaded scheduler in unit tests!
    # For GPU calcs, you will get 'FakeCUDAModule' object has no attribute 'to_device'
    # when attempting numba.jit from multiple threads
    df = ld_matrix(*args, **kwargs).compute(scheduler='single-threaded')
    if not diag:
        df = df.pipe(lambda df: df[df['i'] != df['j']])
    if not nan:
        df = df[~df['value'].isnull()]
    return df

@pytest.mark.parametrize("n", [2, 10, 16, 22])
def test_ld_matrix_window(n):
    # Create zero row vectors except for 1st and 11th 
    # (make them have non-zero variance)
    x = np.zeros((n, 10), dtype='uint8')
    x[0,:-1] = 1
    x[n//2,:-1] = 1
    # All non-self comparisons are nan except for the above two
    df = ldm_df(x, window=n, step=n)
    assert len(df) == 1
    assert df.iloc[0].tolist() == [0, n//2, 1.0]

    # Do the same with physical distance equal to row index
    positions = np.arange(x.shape[0])
    df = ldm_df(x, window=n, positions=positions)
    assert len(df) == 1
    assert df.iloc[0].tolist() == [0, n//2, 1.0]


def test_ld_matrix_threshold():
    # Create zero row vectors except for 1st and 11th 
    # (make them have non-zero variance)
    x = np.zeros((10, 10), dtype='uint8')
    # Make 3rd and 4th perfectly correlated
    x[2,:-1] = 1
    x[3,:-1] = 1
    # Make 8th and 9th partially correlated with 3/4
    x[7,:-5] = 1
    x[8,:-5] = 1
    df = ldm_df(x, window=10)
    # Should be 6 comparisons (2->3,7,8 3->7,8 7->8)
    assert len(df) == 6
    # Only 2->3 and 7->8 are perfectly correlated
    assert len(df[df['value'] == 1.0]) == 2
    # Do the same with a threshold
    df = ldm_df(x, window=10, threshold=.5)
    assert len(df) == 2


@pytest.mark.parametrize("dtype", [
    dtype 
    for k, v in np.sctypes.items() 
    for dtype in v if k in ['int', 'uint', 'float']
])
def test_ld_matrix_dtypes(dtype):
    # Input matrices should work regardless of type
    x = np.zeros((5, 10), dtype=dtype)
    df = ldm_df(x, window=5, diag=True)
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
    return n, m, dict(window=window, step=step, positions=positions, groups=groups)


@given(ld_matrix_args())
@settings(max_examples=50, deadline=None)
def test_ld_matrix_exhaustive_comparisons(args):
    # Validate that no pair-wise comparisons are skipped
    n, m, args = args
    x = np.zeros((n, m), dtype='uint8')
    df = ldm_df(x, diag=True, **args, value_init=-99)
    df_unset = df[df['value'] == -99]
    assert len(df_unset) == 0, \
        f'Found {len(df_unset)} unset values; Examples: {df_unset.head(25)}'
