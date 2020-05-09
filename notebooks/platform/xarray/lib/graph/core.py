"""Graph API"""
from ..dispatch import register_function


DOMAIN = 'graph'


@register_function(DOMAIN)
def maximal_independent_set(df):
    """Maximal Independent Set"""
    pass

