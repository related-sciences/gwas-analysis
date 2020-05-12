from lib.stats.axis_intervals.numba_backend import _axis_intervals as axis_intervals
import numpy as np
import math
import pytest


def interval_union(start, stop):
    """Compute non-unique union of many intervals"""
    return [
        i
        for rng in zip(start, stop)
        for i in range(*rng)
    ]
    
def assert_invariants(ais, cis, n):
    """Conditions that should always be true regardless of paramaterization"""
    assert len(ais) == n
    assert len(cis) <= len(ais)
    # The number of interval members in each interval definition should be the same
    assert ais['count'].sum() == cis['count'].sum()
    # All ranges should be exclusive on the right and non-empty
    assert (cis['max_index'] > cis['min_index']).all()
    assert (cis['max_stop'] > cis['min_start']).all()
    assert (ais['stop'] > ais['start']).all()
    assert (ais['stop'] - ais['start'] == ais['count']).all()
    # All indexes along axis must be within bounds
    def assert_ibi(df, c, n):
        assert df[c].between(0, n).all()
    assert_ibi(ais, 'start', n - 1)
    assert_ibi(cis, 'min_index', n - 1)
    assert_ibi(cis, 'min_start', n - 1)
    assert_ibi(ais, 'stop', n)
    assert_ibi(cis, 'max_index', n)
    assert_ibi(cis, 'max_stop', n)
    # The range of axis elements (usually rows) in both partitionings
    # should have no intersection and be exhaustive
    assert ais['index'].tolist() == list(range(n))
    assert interval_union(cis['min_index'], cis['max_index']) == list(range(n))


def ais_df(n, *args, **kwargs):
    ais, cis = axis_intervals(*args, n=n, **kwargs) 
    ais, cis = ais.to_dataset('var').to_dataframe(), cis.to_dataset('var').to_dataframe()
    assert_invariants(ais, cis, n)
    return ais, cis


@pytest.mark.parametrize("n", [5, 10])
@pytest.mark.parametrize("target_chunk_size", [None, 5, 1])
def test_no_window(n, target_chunk_size):
    # With no window, each interval should only include a single element (typically row)
    ais, cis = ais_df(n=n, window=0, step=None, target_chunk_size=target_chunk_size) 
    assert (ais['count'] == 1).all()
    assert (ais['stop'] - ais['start'] == 1).all()
    assert len(cis) == n / (target_chunk_size or n)

@pytest.mark.parametrize("n", [5, 10])
@pytest.mark.parametrize("target_chunk_size", [None, 5, 1])
def test_unit_window(n, target_chunk_size):
    # With window of 1, each interval should contain one element and its neighbor (to the right)
    ais, cis = ais_df(n=n, window=1, step=None, target_chunk_size=target_chunk_size) 
    # The number of element in each interval should be two except for the last element
    assert (ais['count'].iloc[:-1] == 2).all()
    assert ais['count'].iloc[-1] == 1
    assert (ais['stop'] - ais['start'] <= 2).all()
    # Each chunk should include `target_chunk_size` - 1 elements
    # because inclusion of single neighbors ends each chunk one element earlier
    if target_chunk_size is None:
        assert len(cis) == 1
    elif target_chunk_size == 1:
        assert len(cis) == n
    else:
        assert len(cis) == math.ceil(n / (target_chunk_size - 1))


@pytest.mark.parametrize("target_chunk_size", [None, 5, 1])
def test_window4_step2(target_chunk_size):
    n = 10
    ais, _ = ais_df(n=n, window=4, step=2) 
    # Manually curated example validating the axis intervals
    # (correctness of chunk intervals implied largely by equal sums of these counts)
    assert ais['count'].tolist() == [5, 4, 5, 4, 5, 4, 4, 3, 2, 1]



@pytest.mark.parametrize("step,window", [(0, 1), (-1, 0), (-1, -1), (1, 0), (1, -1), (3, 2)])
def test_raise_on_bad_step_or_window(step, window):
    with pytest.raises(ValueError):
        axis_intervals(n=10, step=step, window=window) 

@pytest.mark.parametrize("target_chunk_size", [None, 3, 1])
def test_window_by_position(target_chunk_size):
    n = 6
    # Separate windows reachable from first, next three, and last two
    positions = np.array([1, 5, 6, 7, 11, 12])
    ais, _ = ais_df(n=n, window=3, positions=positions, target_chunk_size=target_chunk_size) 
    assert ais['count'].tolist() == [1, 3, 2, 1, 2, 1]

@pytest.mark.parametrize("target_chunk_size", [None, 3, 1])
def test_window_by_position_with_groups(target_chunk_size):
    n = 6
    # 1st is on its own, 2nd-4th are within window but broken by group, last two are together
    # Note that position decreses at group break
    positions = np.array([1, 5, 6, 4, 8, 9])
    groups = np.array([1, 1, 1, 2, 2, 2])
    ais, _ = ais_df(n=n, window=3, positions=positions, groups=groups, target_chunk_size=target_chunk_size) 
    assert ais['count'].tolist() == [1, 2, 1, 1, 2, 1]


def test_raise_on_non_monotonic_positions():
        with pytest.raises(ValueError):
            positions = np.array([1, 2, 3, 1, 2, 3])
            axis_intervals(window=1, positions=positions) 

        with pytest.raises(ValueError):
            positions = np.array([3, 2, 1, 3, 2, 1])
            groups = np.array([1, 1, 1, 2, 2, 2])
            axis_intervals(window=1, positions=positions, groups=groups) 