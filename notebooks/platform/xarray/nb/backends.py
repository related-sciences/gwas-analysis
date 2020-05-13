from lib import api
api.config.set('stats.axis_intervals.backend', 'numba')
api.config.set('stats.ld_matrix.backend', 'dask')
api.config.set('graph.maximal_independent_set.backend', 'dask')