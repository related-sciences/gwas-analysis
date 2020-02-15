import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_loadings_traces(df, nbins=96):
    # Stack loadings into long form
    df = df.rename_axis('pc', axis='columns').rename_axis('idx', axis='index') \
        .stack().rename('value').reset_index()
    traces = {}

    # Group by pc and compute binned variant index and loading value counts (we do not want
    # to provide directly to Histogram2D or something of the like since it is too much data)
    for k, g in df.groupby('pc'):
        g = pd.DataFrame(dict(
            value=[b.left for b in pd.cut(g['value'], bins=nbins)],
            idx=[b.left for b in pd.cut(g['idx'], bins=nbins)]
        )).groupby(['value', 'idx']).size().rename('count').reset_index()
        traces[k] = go.Heatmap(x=g['idx'], y=g['value'], z=g['count'], showscale=False)
    return traces


def get_loadings_fig(df, nbins=72, n_pcs=10, n_rows=2, n_cols=5):
    traces = get_loadings_traces(df, nbins=nbins)
    assert n_pcs <= n_rows * n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f'PC{pc + 1} Loadings' for pc in range(n_pcs)])
    for pc in range(n_pcs):
        row, col = int(np.ceil((pc + 1) / n_cols)), (pc % n_cols) + 1
        fig.add_trace(traces[pc], row=row, col=col)
    return fig


def get_loadings_df(ht):
    df = ht.key_by().select('loadings').to_pandas()
    df = pd.DataFrame(df['loadings'].tolist()).reset_index(drop=True)
    return df