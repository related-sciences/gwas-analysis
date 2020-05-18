from lib.method.core import ld_prune
from lib.core import GenotypeCountDataset
from hypothesis import given, example, settings, strategies as st
from hypothesis.extra.numpy import arrays 
import numpy as np
import numpy.testing as npt
import dask
import pytest
import allel
from ..utils import PHASES_NO_SHRINK


@st.composite
def ld_prune_args(draw):
    n_rows, n_cols = draw(st.integers(2, 100)), draw(st.integers(2, 100))
    x = draw(arrays(np.uint8, shape=(n_rows, n_cols), elements=st.integers(0, 2)))
    assert x.ndim == 2
    window = draw(st.integers(1, x.shape[0]))
    step = draw(st.integers(1, window))
    threshold = draw(st.floats(0, 1))
    return x, window, step, threshold

@given(args=ld_prune_args()) # pylint: disable=no-value-for-parameter
@settings(max_examples=50, deadline=None, phases=PHASES_NO_SHRINK)
@example(args=(np.array([[1, 1], [1, 1]], dtype='uint8'), 1, 1, 0.0))
# Ignore warning related to LD calcs for rows with no variance, e.g.
# "RuntimeWarning: invalid value encountered in greater_equal mask = res >= threshold"
@pytest.mark.filterwarnings('ignore:invalid value encountered in greater_equal')
@pytest.mark.parametrize('scheduler', ['single-threaded', 'threads', 'processes'])
def test_vs_skallel(args, scheduler):
    x, window, step, threshold = args
    be_args = dict(backend='numba')

    idx_drop_lib = ld_prune(
        GenotypeCountDataset.create(x),
        window=window,
        step=step,
        threshold=threshold,
        unit='index',
        target_chunk_size=max(x.shape[0]//2, 1),
        ld_matrix_kwargs=dict(backend='dask/numba'),
        axis_intervals_kwargs=be_args, 
        mis_kwargs=be_args
    )
    with dask.config.set(scheduler=scheduler):
        idx_drop_lib = np.sort(idx_drop_lib.index_to_drop.data)
    m = allel.locate_unlinked(x, size=window + 1, step=step, threshold=threshold, blen=x.shape[0])
    idx_drop_ska = np.sort(np.argwhere(~m).squeeze(axis=1))

    npt.assert_equal(idx_drop_ska, idx_drop_lib)