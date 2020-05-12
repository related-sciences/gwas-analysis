DOMAIN = __name__.split('.')[-1]

try:
    from . import pysnptools_backend
except ImportError:
    pass