"""GPU LD prune implementation for tall-skinny arrays with independent set selection via priorities

This method operates in a similar manner to PLINK (--index-pairwise window step), however
variant selection within windows is independent.  In other words, this method will
prune a larger number of variants and without a provided priority, will be equivalent
to choosing a single variant from every connected component in the graph implied by
the supplied LD threshold.  With a priority however, the selection favors high priority
variants that still independent from all other variant selections.

The method works by making comparisons between rows of a call matrix in a given window,
with some configurable number of steps between those windows.  When two rows are found
that exceed some LD threshold, one of the variants associated with a row is eliminated.
This choice can be controlled by providing priorities as `scores` (typically MAF), or it 
will be made arbitrarily, always choosing to keep the first variant in genome order.

As an example, if variant `A` is correlated with `B` and `B` with `C` but not `A` with `C`, 
the first variant in genomic order (`A`) will be chosen and both of the others eliminated.  
If a priority is provided and the priority of `A` and `C` is higher than `B`, both
will be retained.

The time complexity of the method can be summarized as follows:

Assume:
m = num variants
n = num samples
w = window size
s = step size
intsum(x) = x * (x + 1) / 2

Then for all w >= s,
runtime complexity = (intsum(w) - intsum(w - s)) * 2n * (m / s) = O(wnm)
* The worst case is when s = 1
"""
import numpy as np
from numba import cuda
from ....stats import cuda_math as gmath
from ..utils import dotvec1d, eqvec1d
import math

# TODO: Make sure given array is C order (row-major)
# TODO: Need to mean impute missing values (that's what Hail does)
# TODO: Rows need to already be sorted by contig/position (add warning in docs or explicit check?)
# TODO: Check hail vs skallel r2 definitions
# TODO: Preload array into shared memory for comparisons within a thread block

# Use integer ids to select metric functions in kernels
# (passing functions as arguments does not work)
METRIC_IDS = {'r': 1, 'r2': 2, 'dot': 3, 'eq': 4}


def intsum(n):
    """Sum of ints up to `n`"""
    return np.int64((n * (n + 1)) / 2.0) if n > 0 else np.int64(0)

_r = gmath.r
_r2 = gmath.r2
_intsum = cuda.jit(intsum, device=True)
_dot = cuda.jit(dotvec1d, device=True)
_eq = cuda.jit(eqvec1d, device=True)


@cuda.jit(device=True)
def invert_index(i, window, step):
    """Convert truncated squareform index back into row, col, and slice index

    Task indexing for LD pruning is based on several optimizations that utilize a
    cyclic, truncated squareform pattern for pairwise comparisons (between rows).  This pattern
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

    The parameter (`i`) indexes these comparisons where in the above, `i` = 0
    corresponds to the comparison between rows 0 and 1, `i` = 1 to rows 0 and 2, `i` = 4
    to rows 1 and 2, etc.  This method converts this comparison index back into the
    cycle number (arbitrarily called a "slice") as well as offsets within that cycle for the rows
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
        Window size used to define pairwise comparisons
    step : int
        Step size used to define pairwise comparisons

    Returns
    -------
    (i, j, s) : tuple
        i = offset from slice (`s`) to first row in comparison
        j = offset from slice (`s`) to second row in comparison
        s = slice number/index
    """
    assert window >= step
    # Coerce to large float to avoid potential int overflow
    window = np.float64(window)
    step = np.float64(step)

    # Number of pairs in a "slice" = window + (window - 1) + ... + (window - step)
    p = _intsum(window) - _intsum(window - step)

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


@cuda.jit(device=True)
def invert_offset(coords, step):
    """Invert indexing coords to get absolute row indexes"""
    i, j, s = coords
    ri1 = s * step + i
    ri2 = s * step + j
    return ri1, ri2


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
    n_pairs_per_slice = intsum(window) - intsum(window - step)

    # Number of slices in entire array
    n_slices = n_rows // step

    # Determine how many pairwise comparisons are left at the edge of the array
    n_remainder = intsum(window) - intsum(window - (n_rows % step))

    return n_slices * n_pairs_per_slice + n_remainder


@cuda.jit
def _ld_prune_kernel(x, groups, positions, scores, threshold, window, step, max_distance, metric_id, out):
    """LD prune kernel"""

    # Get comparison index for thread and invert to absolute row coordinates
    ci = cuda.grid(1)
    # Note: values here must be unpacked or Numba will complain
    i, j, s = invert_index(ci, window, step)
    ri1, ri2 = invert_offset((i, j, s), step)

    # Ignore if row indexes are OOB (this is expected near final rows)
    if ri1 >= x.shape[0] or ri2 >= x.shape[0]:
        return

    # Only do the comparison if the rows are in the same "group" (e.g. contig)
    if len(groups) > 0 and groups[ri1] != groups[ri2]:
        return

    # Only do the comparison if the rows are within `max_distance` units of
    # each other according to accompanying position vector
    if len(positions) > 0 and 0 < max_distance < abs(positions[ri1] - positions[ri2]):
        return

    # Calculate some metric comparing the rows
    if metric_id == 1:
        v = _r(x[ri1], x[ri2])
    elif metric_id == 2:
        v = _r2(x[ri1], x[ri2])
    elif metric_id == 3:
        v = _dot(x[ri1], x[ri2])
    elif metric_id == 4:
        v = _eq(x[ri1], x[ri2])
    else:
        raise NotImplementedError('Metric not implemented')

    # Debug in sim mode:
    # print('global=', ci, 'i=', i, 'j=', j, 'slice=', s, 'ri1=', ri1, 'ri2=', ri2, 'v=', v)

    # Set boolean indicator based on scores, if metric above threshold
    if v >= threshold:
        # If no scores given, choose to keep earliest row
        if len(scores) == 0:
            out[max(ri1, ri2)] = False
        # Otherwise, choose based on score
        else:
            # Mark the row with the lower associated score as pruned (i.e. not kept)
            if scores[ri1] > scores[ri2]:
                out[ri2] = False
            elif scores[ri2] > scores[ri1]:
                out[ri1] = False
            # If the scores are equal, choose to keep earliest row
            else:
                out[max(ri1, ri2)] = False


def ld_prune(x, window: int, step: int, threshold: float,
             groups=None, positions=None, scores=None,
             metric='r2', max_distance: float = None,
             chunk_offset: int=0):
    """LD prune

    Parameters
    ----------
    x : array (M, N)
        2D array of row vectors to prune
    window : int
        Window size in number of rows
    step : int
        Step size in number of rows
    threshold : float
        Metric threshold (e.g. r2)
    groups : array (M,)
        1D array of row groupings (e.g. contig). Pruning will not occur between rows
        in different groups
    positions : array (M,)
        1D array of global position according to some coordinate system (e.g. genomic coordinate).
        This is only used if `max_distance` is set.  When `groups` is specified as well, the
        effective distance between rows in different groups is infinite so positions only
        need to be specific to one group (e.g. genomic positions along a contig if contigs are groups).
    scores : array (M,)
        1D array of scores used to choose between highly similar rows (e.g. MAF).  For rows
        with metric values
    metric : str
        Name of metric to use for similarity calculations.
        Must be in ['r', 'r2']. Default is 'r2'.  Definitions:
            - r: row correlation in [-1, 1]
            - r2: squared row correlation in [0, 1]
        Note: 'dot' (dot product) and 'eq' (exact equality) are also available for testing/debugging purposes
    max_distance : float
        Maximum distance between rows according to `positions` below which they are still
        eligible for comparison.
    chunk_offset : int
        Offset of first row in `x` within larger chunked (tall-skinny) array.  This is the sum of
        chunk sizes for all chunks prior to the current chunk (i.e. `x`).

    Returns
    -------
    mask : array (M,)
        1D boolean array indicating which rows to keep
    """
    if window < 1:
        raise ValueError(f'Window must be >= 1 (not {window})')
    if step < 1:
        raise ValueError(f'Step must be >= 1 (not {step})')
    if window < step:
        raise ValueError(f'Window must be >= step (not window={window}, step={step})')
    if metric not in METRIC_IDS:
        raise ValueError(f'Metric must be in {list(METRIC_IDS.keys())} (not {metric})')

    # Number of rows in columns in data
    nr, nc = x.shape

    # Set argument defaults for GPU
    # Use default arrays such that len(a) == 0, this is
    # sentinel condition available on gpu (None doesn't work)
    if max_distance is None:
        max_distance = 0
    if groups is None:
        groups = np.empty(0)
    if positions is None:
        positions = np.empty(0)
    if scores is None:
        scores = np.empty(0)

    # Move necessary data to GPU
    x = cuda.to_device(x)
    groups = cuda.to_device(groups)
    positions = cuda.to_device(positions)
    scores = cuda.to_device(scores)

    # Output is num rows true/false vector where true = keep row
    out = cuda.to_device(np.ones(nr, dtype='bool'))

    # Determine number of pairwise row comparisons necessary
    n_tasks = num_comparisons(nr, window, step)

    kernel = _ld_prune_kernel
    kernel.forall(n_tasks)(
        x, groups=groups, positions=positions, scores=scores, threshold=threshold,
        window=window, step=step, max_distance=max_distance, metric_id=METRIC_IDS[metric],
        out=out
    )

    return out.copy_to_host()
