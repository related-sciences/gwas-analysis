"""Package-wide configuration options and management

TODO: Start discussion on configuration from filesystem
TODO: Extract pandas private data structures or make our own for this
"""
import pandas as pd
from pandas._config.config import register_option as register_opt


def describe_option(pat="", _print_desc=True):
    return pd.describe_option(pat, _print_desc)


def option_context(*args):
    return pd.option_context(*args)


def register_option(key, defval, doc="", validator=None, cb=None):
    # Lift from https://github.com/pandas-dev/pandas/blob/aa1089f5568927bd660a6b5d824b20d456703929/pandas/_config/config.py#L423
    register_opt(key, defval, doc, validator, cb)


def get_option(key):
    return pd.get_option(key)


def set_option(*args, **kwargs):
    pd.set_option(*args, **kwargs)
