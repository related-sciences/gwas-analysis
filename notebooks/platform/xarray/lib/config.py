"""Package-wide configuration options and management

TODO: Start discussion on configuration from filesystem
TODO: Extract pandas private data structures or make our own for this
"""
import pandas as pd
from pandas._config.config import register_option as register_option
from pandas._config.config import _registered_options as registered_options
from pandas._config.config import OptionError


# This is currently a temporary wrapper around pandas options, but should
# eventually inherit from ABC types.  For now, it's unclear to me what the
# best way to do this is with attributes to hang documentation off of
# as well as context managers (which are both helpful)
# TODO: subclass ABC and make non-global
class Configuration:

    def __getitem__(self, key):
        return pd.get_option(key)

    def __getattr__(self, item):
        return getattr(pd.options, item)

    def __contains__(self, item):
        try:
            pd.get_option(item)
            return True
        except OptionError:
            return False

    def get(self, key, default=None):
        return self[key] if key in self else default

    def set(self, key, value):
        pd.set_option(key, value)

    def describe(self, pat=""):
        return pd.describe_option(pat)

    def context(self, *args):
        return pd.option_context(*args)

    def register(self, key, default_value, doc=""):
        if key in registered_options:
            del registered_options[key]
        # Lift from https://github.com/pandas-dev/pandas/blob/aa1089f5568927bd660a6b5d824b20d456703929/pandas/_config/config.py#L423
        register_option(key, default_value, doc)

    def __setattr__(self, key, value):
        pd.set_option(key, value)

config = Configuration()
