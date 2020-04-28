"""GPU LD prune implementation for tall-skinny arrays"""
import numpy as np
from numba import cuda
import math


def _intsum(n):
    """Sum of ints up to `n`"""
    return np.int64((n * (n + 1)) / 2.0) if n > 0 else np.int64(0)


_intsumd = cuda.jit(_intsum, device=True)


@cuda.jit(device=True)
def invert_index(i, window, step):
    """Convert truncated squareform index back into row, col, and slice index

    Task indexing for LD pruning is based on several optimizations that utilize a
    cyclic, truncated squareform pattern for pair-wise comparisons (between rows).  This pattern
    is primarily   controlled by window and step parameters, where an example for window = 4 and
    step = 3 would look like this:

    row index  row indexes of other rows to compare to
            |   |
            0 | 1 2 3 4
            1 | 2 3 4
            2 | 3 4
            3 | 4 5 6 7
            4 | 5 6 7
            5 | 6 7
            6 | 7 8 9 10
            ... and so on ...

    The comparison index parameter (`i`) indexes these comparisons where in the above, `i` = 0
    corresponds to the comparison between rows 0 and 1, `i` = 1 to rows 0 and 2, `i` = 4
    to rows 1 and 2, etc.  This method converts this comparison index back into the
    cycle number (called a "slice") as well as offsets within that cycle for the rows
    being compared.  The slice number itself indexes some row in the original array
    and the offsets can be used to identify comparisons from that row index.

    Examples for the same case above for given comparison index values are:

    index -> (row, col, slice)
    0 -> (0, 1, 0) -
    1 -> (0, 2, 0) |
    2 -> (0, 3, 0) |
    3 -> (0, 4, 0) |
    4 -> (1, 2, 0) |--> One "slice" (i.e. one cycle)
    5 -> (1, 3, 0) |
    6 -> (1, 4, 0) |
    7 -> (2, 3, 0) |
    8 -> (2, 4, 0) -
    9 -> (0, 1, 1) # The pattern repeats here

    Parameters
    ----------
    i : int
        Comparison index
    window : int
        Window size used to define pair-wise comparisons
    step : int
        Step size used to define pair-wise comparisons

    Returns
    -------
    (i, j, s) : tuple
        i = offset from slice (`s`) to first row in comparison
        j = offset from slice (`s`) to second row in comparison
        s = slice number/index
    """
    assert window >= step
    window = np.float64(window)
    step = np.float64(step)

    # Number of pairs in a "slice" = window + (window - 1) + ... + (window - step)
    p = _intsumd(window) - _intsumd(window - step)

    # Calculate slice number (`s`) and offset into that slice (`k`)
    s, k = np.int64(i // p), np.int64(i % p)

    # Invert squareform index
    # See: https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    n = window + 1  # The "n" in this case is the size of the window + 1 since self comparisons are ignored
    i = np.int64(n - 2 - math.floor(math.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5))
    j = np.int64(k + i + 1 - n * (n - 1) / 2.0 + (n - i) * ((n - i) - 1) / 2.0)

    assert i >= 0
    assert j >= 0
    assert s >= 0
    return i, j, s


def num_comparisons(n_rows, window, step):
    """Get number of comparisons implied by array size and parameters

    Note that this total will count comparisons near the end of an array
    that aren't actually possible (e.g. if the final row is at the start
    of a slice, all `window` comparisons implied are added to total
    even if they would be OOB).  It is easier to simply ignore these
    cases during calculation than it is to subtract them out of the
    indexing scheme.
    """

    # Number of pairs in each slice
    n_pairs_per_slice = _intsum(window) - _intsum(window - step)

    # Number of slices in entire array
    n_slices = n_rows // step

    # Determine how many pairwise comparisons are left at the edge of the array
    n_remainder = _intsum(window) - _intsum(window - (n_rows % step))

    return n_slices * n_pairs_per_slice + n_remainder


@cuda.jit(device=True)
def invert_offset(coords, step):
    """Invert indexing coords to get absolute row indexes"""
    i, j, s = coords
    ri1 = s * step + i
    ri2 = s * step + j
    return ri1, ri2


@cuda.jit
def _ld_prune_kernel(x, groups, positions, scores, threshold, window, step, max_distance, out):
    """LD prune kernel"""

    # Get comparison index for thread and invert to absolute row coordinates
    ci = cuda.grid(1)
    i, j, s = invert_index(ci, window, step)
    ri1, ri2 = invert_offset((i, j, s), step)

    # Ignore if row indexes are OOB (this is expected near final rows)
    if ri1 >= x.shape[0] or ri2 >= x.shape[0]:
        return

    # Only do the comparison if the rows are in the same "group" (e.g. contig)
    if groups[ri1] != groups[ri2]:
        return

    # Calculate some metric comparing the rows (inner product as a test)
    v = 0.0
    for k in range(x.shape[1]):
        v += x[ri1, k] * x[ri2, k]

    # i, j, s = coords
    # print('global=', ci, 'i=', i, 'j=', j, 'slice=', s, 'ri1=', ri1, 'ri2=', ri2, 'v=', v)

    # Set boolean indicator based on scores, if metric above threshold
    if v >= threshold:
        # Mark the row with the lower associated score as pruned (i.e. not kept)
        if scores[ri1] > scores[ri2]:
            out[ri2] = False
        elif scores[ri2] > scores[ri1]:
            out[ri1] = False
        # If the scores are equal, ignore the row with the highest index
        else:
            out[max(ri1, ri2)] = False


def ld_prune_gpu(x, groups, positions, threshold: float, window: int, step: int,
                 metric='r2', scores=None, max_distance: float = None, chunk_offset: int=0):
    """LD prune

    Parameters
    ----------
    x : array (n_rows, n_cols)
        2D array of row vectors to prune
    groups : array (n_rows)
        1D array of row groupings (e.g. contig). Pruning will not occur between rows
        in different groups
    positions : array (n_rows)
        1D array of global position according to some coordinate system (e.g. genomic coordinate).
        This is only used if `max_distance` is set.
    threshold : float
        Metric threshold (e.g. r2)
    window : int
        Window size in number of rows
    step : int
        Step size in number of rows
    metric : str
        Name of metric to use for similarity calculations (only 'r2' at this time)
    scores : array
        1D array of scores used to choose between highly similar rows (e.g. MAF).  For rows
        with metric values
    max_distance : float
        Maximimum distance between rows according to `position` below which they are still
        eligible for comparison
    chunk_offset : int
        Offset of first row in `x` within larger chunked (tall-skinny) array.  This is the sum of
        chunk sizes for all chunks prior to the current chunk (i.e. `x`).

    Returns
    -------
    mask: array (n_rows)
        1D boolean array indicating which rows to keep
    """
    assert window > 0
    assert step > 0
    assert window >= step

    # Number of rows in columns in data
    nr, nc = x.shape

    # Set argument defaults for GPU
    if max_distance is None:
        max_distance = 0
    if positions is None:
        positions = np.empty(1)
    if scores is None:
        scores = np.zeros(nr, dtype=np.int8)

    # Move necessary data to GPU
    x = cuda.to_device(x)
    groups = cuda.to_device(groups)
    positions = cuda.to_device(positions)
    if scores is not None:
        scores = cuda.to_device(scores)

    # Output is num rows true/false vector where true = keep row
    out = cuda.to_device(np.ones(nr, dtype='bool'))

    n_tasks = num_comparisons(nr, window, step)
    print('max_index:', n_tasks)
    kernel = _ld_prune_kernel
    kernel.forall(n_tasks)(x, groups, positions, scores, threshold, window, step, max_distance, out)

    return out.copy_to_host()
