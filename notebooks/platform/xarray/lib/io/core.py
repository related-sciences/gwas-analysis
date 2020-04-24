"""I/O backend implementations and configuration"""
from ..dispatch import FrontendDispatcher, ClassBackend, Domain

DOMAIN = Domain('io')
PLINK_DOMAIN = DOMAIN.append('plink')

# ----------------------------------------------------------------------
# IO API


class IOBackend(ClassBackend):
    domain = DOMAIN


class PLINKBackend(IOBackend):
    domain = PLINK_DOMAIN


dispatchers = dict()


def dispatch(domain):
    if domain not in dispatchers:
        dispatchers[domain] = FrontendDispatcher(DOMAIN.append(domain))

    def decorator(fn):
        return dispatchers[domain].add(fn)
    return decorator


@dispatch('plink')
def read_plink(path, backend=None, **kwargs):
    """Import PLINK dataset"""
    pass


# ----------------------------------------------------------------------
# IO Backend Registration


def register_backend(backend: IOBackend):
    dispatchers[backend.domain[-1]].register(backend)
