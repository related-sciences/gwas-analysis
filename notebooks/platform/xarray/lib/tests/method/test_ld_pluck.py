from lib.method.ld_pluck.cuda_backend import invert_index, invert_offset, num_comparisons, ld_pluck
import numpy as np
import functools

def test_invert_index(self):
    for case in INDEX_CASES:
        w, s = case['window'], case['step']
        for c in case['indexes']:
            i, expected = c
            actual = invert_index(i, w, s)
            assert actual == expected

            def test_period_increment(case, inc, expected):
                # Move index forward by `inc` periods and assert that results do not change
                # e.g. if there are 9 comparisons in a slice, check that index 0 has same result as index 9
                actual = invert_index(i + inc * (len(case['indexes']) - 1), w, s)
                # Add inc to slice number since we have intentionally increased this
                expected = expected[:-1] + (expected[-1] + inc,)
                assert actual == expected

            # Test small period shifts in index being inverted
            test_period_increment(case, 1, tuple(expected))
            test_period_increment(case, 2, tuple(expected))

            # Test large period shift
            test_period_increment(case, 20000, tuple(expected))

            # Test shift beyond int32 max
            test_period_increment(case, int(1E12), tuple(expected))

def test_num_comparisons(self):
    for case in INDEX_CASES:
        n_pairs_per_slice = len(case['indexes']) - 1
        step = case['step']
        window = case['window']

        # Num rows equal to step size
        n_rows = step
        expected = n_pairs_per_slice
        actual = num_comparisons(n_rows, window, step)
        assert actual == expected

        # Num rows double step size
        n_rows = step * 2
        expected = n_pairs_per_slice * 2
        actual = num_comparisons(n_rows, window, step)
        assert actual == expected

        # One row beyond step size
        n_rows = step + 1
        expected = n_pairs_per_slice + window
        actual = num_comparisons(n_rows, window, step)
        assert actual == expected

        # Num rows double step size with two extra
        n_rows = step * 2 + 2
        expected = n_pairs_per_slice * 2 + window
        # For step size 1, each row is its own slice so
        # the remainder does not result in a different
        # number of extra comparisons
        if step > 1:
            expected += window - 1
        else:
            expected += window
        actual = num_comparisons(n_rows, window, step)
        assert actual == expected

def test_invert_offset():
    def get_rows(ci, window, step):
        return invert_offset(invert_index(ci, window, step), step)

    window, step = 1, 1
    assert get_rows(0, window, step) == (0, 1)
    assert get_rows(1, window, step) == (1, 2)
    assert get_rows(2, window, step) == (2, 3)

    window, step = 3, 1
    assert get_rows(0, window, step) == (0, 1)
    assert get_rows(1, window, step) == (0, 2)
    assert get_rows(2, window, step) == (0, 3)
    assert get_rows(3, window, step) == (1, 2)

    window, step = 4, 3
    # ci = 0 => slice = 0 => row indexes start at 0
    assert get_rows(0, window, step) == (0, 1)
    # ci = 9 => slice = 1 (first time windows moves `step`) => row indexes start at 3
    assert get_rows(9, window, step) == (3, 4)
    assert get_rows(10, window, step) == (3, 5)
    assert get_rows(12, window, step) == (3, 7)
    assert get_rows(13, window, step) == (4, 5)

    window, step = 4, 2
    # ci = 0 => slice = 0 => row indexes start at 0
    assert get_rows(7, window, step) == (2, 3)
    assert get_rows(8, window, step) == (2, 4)
    assert get_rows(10, window, step) == (2, 6)
    assert get_rows(11, window, step) == (3, 4)


def test_ld_pluck_groups():
    run = functools.partial(ld_pluck, window=3, step=1, threshold=1, positions=None)
    # 3 vectors, none eliminated despite being equal b/c
    # they're in different groups
    x = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    groups = np.array([1, 2, 3])
    res = run(x, groups=groups, metric='eq')
    assert np.all(res)

def test_ld_pluck_selection():
    run = functools.partial(ld_pluck, step=1, threshold=1, positions=None)

    # 3 vectors, none eliminated
    x = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    res = run(x, window=1, metric='eq', groups=None)
    assert np.all(res)
    res = run(x, window=3, metric='eq', groups=None)
    assert np.all(res)

    # Larger array test for last rows that get pruned
    x = np.zeros((1_000, 10), dtype='uint8')
    # all rows are zero except for the last two, which are all one
    x[-2:, :] = 1
    res = run(x, window=1, metric='dot', groups=None)
    # Only the final row should be marked as pruned
    assert np.all(res[:-1])
    assert not res[-1]

    # Test when all rows should be pruned (except first)
    x = np.zeros((1_000, 10), dtype='uint8')
    res = run(x, window=1, metric='eq', groups=None)
    assert np.all(~res[1:])
    assert res[0]

def test_ld_pluck_r2():
    run = functools.partial(ld_pluck, step=3, window=5, groups=None, positions=None)

    # Validate that when rows are unique, no correlation is >= 1
    x = unique_row_array((100, 10), 'uint8')
    # Note that 'r' is used rather than 'r2' because there can
    # be unique pairs of rows with perfect negative correlation
    res = run(x, metric='r', threshold=1)
    assert np.all(res)

    # Validate that when rows are identical, all correlation is >= 1
    x = np.zeros((100, 10), 'uint8')
    # Make the last column all ones so each row has non-zero variance
    x[:, -1] = 1
    res = run(x, metric='r2', threshold=1)
    # Every row but first should be pruned
    assert np.all(~res[1:])
    assert res[0]


def unique_row_array(shape, dtype):
    """Generate an array with unique rows"""
    x = np.zeros(shape, dtype)
    for i in range(x.shape[0]):
        fmt = '{0:0' + str(x.shape[1]) + 'b}'
        bits = list(fmt.format(i))
        assert len(bits) == x.shape[1], \
            f'Not enough columns present ({x.shape[1]}) to create {x.shape[0]} unique rows'
        # Make row vector equal to bit representation of row index integer
        x[i] = np.array(bits).astype(dtype)
    return x


INDEX_CASES = [
    # Window = step + 1
    dict(
        window=4, step=3,
        indexes=[
            (0, (0, 1, 0)),
            (1, (0, 2, 0)),
            (2, (0, 3, 0)),
            (3, (0, 4, 0)),
            (4, (1, 2, 0)),
            (5, (1, 3, 0)),
            (6, (1, 4, 0)),
            (7, (2, 3, 0)),
            (8, (2, 4, 0)),
            (9, (0, 1, 1)),
        ]
    ),
    # Window > step + 1
    dict(
        window=5, step=1,
        indexes=[
            (0, (0, 1, 0)),
            (1, (0, 2, 0)),
            (2, (0, 3, 0)),
            (3, (0, 4, 0)),
            (4, (0, 5, 0)),
            (5, (0, 1, 1)),
        ]
    ),
    # Window = step but > 1
    dict(
        window=3, step=3,
        indexes=[
            (0, (0, 1, 0)),
            (1, (0, 2, 0)),
            (2, (0, 3, 0)),
            (3, (1, 2, 0)),
            (4, (1, 3, 0)),
            (5, (2, 3, 0)),
            (6, (0, 1, 1)),
        ]
    ),
    # Window = step = 1
    dict(
        window=1, step=1,
        indexes=[
            (0, (0, 1, 0)),
            (1, (0, 1, 1)),
        ]
    ),
]