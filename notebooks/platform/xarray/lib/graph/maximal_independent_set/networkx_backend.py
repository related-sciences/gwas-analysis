import networkx as nx
from xarray import Dataset
from ...typing import DataMapping
from ...dispatch import dispatches_from, NetworkxBackend
from .. import core


def _maximal_independent_set(idi, idj, cmp):
    G = nx.Graph()
    G.add_edges_from(zip(idi, idj))
    index_to_keep = nx.algorithms.mis.maximal_independent_set(G)
    return list(set(G.nodes) - set(index_to_keep))

@dispatches_from(core.maximal_independent_set)
def maximal_independent_set(ds: DataMapping) -> DataMapping:
    """Networkx MIS

    Note: `cmp` is ignored by this backend.

    Returns
    -------
    Dataset
    """
    return core.wrap_mis_vec_fn(_maximal_independent_set)(ds)

core.register_backend_function(NetworkxBackend)(maximal_independent_set)