from abc import abstractmethod
from ..dispatch import ClassBackend
from ..compat import dask_dataframe_type, pandas_dataframe_type
from multipledispatch import dispatch
from functools import partial
import pandas as pd
import numpy as np

dispatch = partial(dispatch, namespace=dict())


class MISBackend(ClassBackend):

    @abstractmethod
    def run(self, idi, idj, cmp):
        pass 

    @dispatch(pandas_dataframe_type)
    def maximal_independent_set(self, df):
        def get(c):
            return np.asarray(df[c].values) if c in df else None
        drop = self.run(*[get(c) for c in ['i', 'j', 'cmp']])
        return pd.DataFrame({'index_to_drop': list(drop)})

    # pylint: disable=function-redefined
    @dispatch(dask_dataframe_type)
    def maximal_independent_set(self, df):
        import dask.dataframe as dd
        return dd.from_pandas(self.maximal_independent_set(df.compute()))
