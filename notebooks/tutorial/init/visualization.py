import plotnine as pn
import numpy as np


def manhattan_plot(df, limit=20000):
    return (
        pn.ggplot(
            df
                .sort_values('P')
                .reset_index(drop=True)
                .head(limit)
                .assign(LOGP=lambda df: -np.log10(df['P']))
                .assign(CHR=lambda df: df['CHR'].astype(str))
            ,
            pn.aes(x='POS', y='LOGP', fill='CHR', color='CHR')
        ) + 
        pn.geom_point() + 
        pn.geom_hline(yintercept=5) + 
        pn.theme_bw() + 
        pn.theme(figure_size=(16, 4))
    )

def qq_plot(df, limit=20000):
    return (
        pn.ggplot(
            df
                .sort_values('P')
                .assign(OBS=lambda df: -np.log10(df['P']))
                .assign(EXP=lambda df: -np.log10(np.arange(1, len(df) + 1) / float(len(df))))
                .head(limit),
            pn.aes(x='EXP', y='OBS')
        ) + 
        pn.geom_point() + 
        pn.geom_abline() + 
        pn.theme_bw() 
    )