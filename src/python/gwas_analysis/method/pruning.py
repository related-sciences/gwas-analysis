from dataclasses import dataclass
from numba import jit
from typing import List

import dask.array as da
import numpy as np


##############################
# Contig/Chunk Model Classes #
##############################
# These are necessary for aligning array chunks to contig splits


@dataclass(frozen=True)
class ChunkContigInfo:
    contig_index: int
    contig_value: int
    chunk_idx: List[int]
    chunk_size: List[int]


@dataclass(frozen=True)
class ChunkInfo:
    chunks: List[ChunkContigInfo]

    def get_contig_chunk_boundary(self):
        """ Get index in last chunk for each contig keyed by chunk index """
        bounds = {}
        for c in self.chunks:
            bounds[c.chunk_idx[-1]] = c.chunk_size[-1]
        return bounds

    def get_chunk_offset(self):
        """ Get global offset for first row index in each chunk keyed by chunk index """
        offsets = {}
        o = 0
        for c in self.chunks:
            for i, s in zip(c.chunk_idx, c.chunk_size):
                offsets[i] = o
                o += s
        return offsets


def get_chunk_info(pos, size):
    chunks = []
    csct = 0
    for i, (v, c) in enumerate(zip(*np.unique(pos[:, 0], return_counts=True))):
        sizes = [size] * (c // size)
        if c % size > 0:
            sizes += [int(c % size)]
        idx = [j + csct for j in range(len(sizes))]
        csct += len(sizes)
        chunks.append(
            ChunkContigInfo(
                contig_index=i, contig_value=v, chunk_idx=idx, chunk_size=sizes
            )
        )
    return ChunkInfo(chunks)


#####################
# Numba/Numpy Shims #
#####################
# Workarounds for lack of per-axis reduction in numba
# See: https://github.com/numba/numba/issues/1269


@jit(nopython=True, nogil=True)
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@jit(nopython=True, nogil=True)
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@jit(nopython=True, nogil=True)
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)


@jit(nopython=True, nogil=True)
def np_argmin(array, axis):
    return np_apply_along_axis(np.argmin, axis, array)


#####################
# Pruning Functions #
#####################
# These functions define a parallel, block-wise LD prune algorithm that first
# aligns array chunks to contigs and step/window intervals so that overlapping
# calculations are not out of phase across chunks


@jit(nopython=True, nogil=True)
def corrcoef(gn0, gn1, gn0_sq, gn1_sq):
    # initialise variables
    m0 = m1 = v0 = v1 = cov = n = 0

    # iterate over input vectors
    for i in range(gn0.shape[0]):
        x = gn0[i]
        y = gn1[i]
        # consider negative values as missing
        if x >= 0 and y >= 0:
            n += 1
            m0 += x
            m1 += y
            xsq = gn0_sq[i]
            ysq = gn1_sq[i]
            v0 += xsq
            v1 += ysq
            cov += x * y

    # early out
    if n == 0 or v0 == 0 or v1 == 0:
        return np.nan

    # compute mean, variance, covariance
    m0 /= n
    m1 /= n
    v0 /= n
    v1 /= n
    cov /= n
    cov -= m0 * m1
    v0 -= m0 * m0
    v1 -= m1 * m1

    # compute correlation coeficient
    r = cov / np.sqrt(v0 * v1)
    return r


def _prune(
    X,
    block_id=None,
    window=None,
    step=None,
    threshold=None,
    contig_boundary=None,
    chunk_offset=None,
    overlap_depth=None,
    short_circuit=False,
    short_circuit_step_rate=2,
    return_ld_matrix=False,
):
    assert block_id is not None
    assert window is not None
    assert step is not None
    assert threshold is not None
    assert contig_boundary is not None
    assert chunk_offset is not None
    assert overlap_depth is not None

    EPS = 1e-6

    # Always eliminate leading padding rows
    X = X[overlap_depth:]
    # Eliminate padding rows if in last chunk and
    # determine global offset from original array
    # if block_id[0] == max(chunk_offset.keys()):
    #     X = X[:-overlap_depth]
    # row_offset = chunk_offset.get(block_id[0])
    row_offset = None
    max_bid = 0
    for (bid, offset) in chunk_offset:
        if bid > max_bid:
            max_bid = bid
        if block_id[0] == bid:
            row_offset = offset
    if block_id[0] == max_bid:
        X = X[:-overlap_depth]
    assert row_offset is not None

    # Determine max row index for contig (only applies to overlap)
    # row_max = contig_boundary.get(block_id[0])
    for (bid, rmax) in contig_boundary:
        if block_id[0] == bid:
            X = X[:rmax]
            break

    # Run preprocessing for triangle inequality short-circuiting
    if short_circuit:
        # TODO: This needs to be nan-insensitive
        Xc = (X - np.expand_dims(np_mean(X, 1), 1)) / np.expand_dims(np_std(X, 1), 1)
        Xs = Xc[:: (step // short_circuit_step_rate)]
        # Make sure to divide dot products by number of columns
        # TODO: Transpose is no longer row major and matmul throws numba perf warning
        # so presumably the cost of a copy to C contiguous array is worth it here --
        # this may be worth some benchmarking to see if the warning should just be ignored instead
        Xs = (Xc @ np.ascontiguousarray(Xs.T)) / Xc.shape[1]
        # Convert correlation to distance (d = 1 - corr => corr = 1 - d)
        Xs = 1 - Xs
        assert np.all((Xs >= -EPS) & (Xs <= 2 + EPS))
        Xsi = np_argmin(Xs, 1).astype(np.int64)
        assert Xsi.shape[0] == Xs.shape[0] == X.shape[0]

    n, m = X.shape
    if return_ld_matrix:
        ldm = np.ones((n, n), dtype=np.float32) * -2
    keep = np.ones(n, dtype=np.uint8)
    # Loop over window start index
    for w_start in range(0, n, step):
        w_stop = min(w_start + window, n)
        # Loop over primary row index
        for i in range(w_start, w_stop):
            if not keep[i]:
                continue
            # Loop over secondary row index, noting that every step moved
            # since the beginning of the last window decreases the number
            # of overlapping columns in the correlation matrix already
            # calculated (so they can be skipped to avoid recomputation)
            j_start = i + 1 if w_start == 0 else max(i + 1, w_start + window - step)
            for j in range(j_start, w_stop):
                if not keep[j]:
                    continue
                if short_circuit:
                    # Find closest vector to primary
                    min_dist_idx = Xsi[i]
                    di = Xs[i, min_dist_idx]
                    dj = Xs[j, min_dist_idx]
                    # Determine distance lower bound to secondary
                    dlb = abs(di - dj)
                    # Convert back to r2 and continue if we can be sure these
                    # two rows are sufficiently uncorrelated
                    cub = 1 - dlb
                    if cub ** 2 <= threshold:
                        continue
                xi, xj = X[i], X[j]
                r2 = corrcoef(xi, xj, xi ** 2, xj ** 2) ** 2
                # mask = (xi >= 0) & (xj >= 0) # Ignore missing
                # xi, xj = xi[mask], xj[mask]
                # r2 = np.corrcoef(xi, xj)[0, 1] ** 2 # This ends up being WAY slower

                # Small rounding errors like 1.0000000000000013 do occur
                assert np.isnan(r2) or (-1 - EPS <= r2 <= 1 + EPS)
                if return_ld_matrix:
                    ldm[i, j] = r2
                if r2 > threshold:
                    keep[j] = False

    # Return types must be the same for jit even if results contain different quantities
    if return_ld_matrix:
        ldmr = ldm.reshape(-1)
        blk_ids_lng = np.repeat(block_id[0], repeats=len(ldmr)).astype(ldmr.dtype)
        return np.stack((ldmr, blk_ids_lng), axis=1)
    else:
        keep_idx = np.argwhere(keep)[:, 0].astype(np.float32) + row_offset
        blk_ids = np.repeat(block_id[0], repeats=len(keep_idx)).astype(np.float32)
        return np.stack((keep_idx, blk_ids), axis=1)


def prune(
    X,
    pos,
    window,
    step,
    threshold,
    align_chunks=True,
    short_circuit=False,
    windows_per_chunk=10,
    numba=False,
    compute=True,
):
    assert step < window
    assert window % step == 0
    assert X.ndim == 2

    # Gather information defining how the array should be chunked
    chunk_size = windows_per_chunk * window
    chunk_info = get_chunk_info(pos, chunk_size)
    # Create list of individual chunk sizes, which will almost always be uneven
    # due to breaks at contig boundaries
    chunk_lens = tuple(cs for ci in chunk_info.chunks for cs in ci.chunk_size)

    # Rechunk the provided array to match assumptions made by this method
    if not align_chunks and X.chunks[0] != chunk_lens:
        raise ValueError(f"Expected chunks {chunk_lens}, found {X.chunks[0]}")
    if align_chunks:
        X = X.rechunk(chunks=(chunk_lens, X.chunks[1]))

    # Determine how many rows should be shared in each block calculation
    overlap_depth = window - step

    fn = jit(_prune, nopython=True, nogil=True) if numba else _prune
    R = da.map_overlap(
        X,
        fn,
        window=window,
        step=step,
        overlap_depth=overlap_depth,
        depth=(overlap_depth, 0),  # No overlap in axis 1
        threshold=threshold,
        boundary=-1,
        short_circuit=short_circuit,
        # Use tuples because numbda dict results in this when pickled for serialized tasks:
        # TypeError: can't pickle _nrt_python._MemInfo objects"
        contig_boundary=tuple(chunk_info.get_contig_chunk_boundary().items()),
        chunk_offset=tuple(chunk_info.get_chunk_offset().items()),
        short_circuit_step_rate=2,
        # TODO: What are the consequences of providing chunk row size that is <= result here?
        chunks=([v for v in X.chunks[0]], 2),
        dtype=np.float64,
        trim=False,
        return_ld_matrix=False,
    )

    # Result contains rows indices to keep in first column (block index in second)
    if compute:
        R = R.compute()
        Xp = X[np.unique(R[:, 0]).astype(int)]
        return Xp, (X, R, chunk_info, overlap_depth)
    return R, (X, chunk_info, overlap_depth)
