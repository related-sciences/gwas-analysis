## Xarray Genetic Data API Prototype

This prototype library proposes a data model and API for genetic data.  These implementations should be largely compatible with the [skallel](https://github.com/scikit-allel/skallel) framework and this is intended as a starting point on a top-down design.

Links:

### Notebooks

- [io.ipynb](https://nbviewer.jupyter.org/github/related-sciences/gwas-analysis/blob/master/notebooks/platform/xarray/io.ipynb) - IO plugin framework, export optimizations, and PLINK examples
- [dispatching.ipynb](https://nbviewer.jupyter.org/github/related-sciences/gwas-analysis/blob/master/notebooks/platform/xarray/dispatching.ipynb) - A short discussion on problems with dispatching to pydata array backends and proposed solutions (the IO example above uses the framework illustrated here)
- [data_structures.ipynb](https://nbviewer.jupyter.org/github/related-sciences/gwas-analysis/blob/master/notebooks/platform/xarray/data_structures.ipynb) - Xarray Dataset subtype composition and accessors
- [indexing.ipynb](https://nbviewer.jupyter.org/github/related-sciences/gwas-analysis/blob/master/notebooks/platform/xarray/indexing.ipynb) - Xarray indexing, selection, and "join" examples
- [qc_call_rate_benchmarking.ipynb](https://nbviewer.jupyter.org/github/related-sciences/gwas-analysis/blob/master/notebooks/platform/xarray/qc_call_rate_benchmarking.ipynb) - Benchmarking of simple QC operations for comparison to Hail, Glow and PLINK
- Note: All notebooks here don't render well on github so links point to nbviewer (use that for browsing them or they're difficult to understand)

### Code 

- The internals for data structures are in [core.py](lib/core.py)
- The proposed dispatching framework is in [dispatch.py](lib/dispatch.py)
- An example PLINK backend is in [io/pysnptools_backend.py](lib/io/pysnptools_backend.py)
- The custom numcodec filter used for bitpacking diploid, bi-allelic data is in [io/codecs.py](lib/io/codecs.py)

 