from .core import (
    GeneticDataset,
    GenotypeAlleleCountDataset,
    GenotypeCallDataset,
    GenotypeCountDataset,
    GenotypeDosageDataset,
    GenotypeProbabilityDataset
)

from .config import config
from . import dispatch

from . import io
from .io.core import (
    read_bgen,
    read_plink,
    write_zarr
)

from . import stats
from .stats.core import (
    ld_matrix,
    axis_intervals
)

from . import method
from .method.core import (
    ld_prune
)

from . import graph
from .graph.core import (
    maximal_independent_set
)

config.register('variable.groups', 'contig')
config.register('variable.positions', 'pos')