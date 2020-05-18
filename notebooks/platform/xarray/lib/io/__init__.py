DOMAIN = __name__.split('.')[-1]

try:
    from . import pysnptools_backend
except ImportError:
    pass

try:
    from . import codecs
except ImportError:
    pass

try:
    from . import plugins
except ImportError:
    pass