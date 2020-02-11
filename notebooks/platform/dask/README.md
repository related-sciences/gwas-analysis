## Dask Gentotype Data QC Prototype

The [Dask Prototype](prototype.ipynb) notebook in this directory demonstrates the efficiency of a single genetic data QC workflow step using Dask, Numpy, and Zarr.  This step was chosen because it was the subject of performance discussions relating to how slow the same step is when executed via Glow or Hail (over Spark).  The relevant Glow discussion is [here](https://github.com/projectglow/glow/issues/148) and the Hail discussion [here](https://discuss.hail.is/t/poor-performance-for-qc-filtering-on-medium-sized-genotype-data/1263/21).

This particular workflow step involves nothing more than sequential applications of row and column mean calculations, followed by appropriate filtering.  On 1000 Genomes data, a 16G ```25488488 x 629``` matrix (when modeled as uint8 alternate allele counts), this step takes ~16m with Hail and ~1h25m with Glow.  It takes only 35 seconds with Dask, which is nearly identical to the time taken by PLINK (albeit on 16 cores instead of 1).

This prototype involves two pieces of custom functionality:

- A PLINK file reader with support for slicing/file seeks provided by [pysnptools](https://github.com/fastlmm/pysnptools)
- A Zarr codec that implements the same bitpacking scheme as the PLINK [.bed](https://www.cog-genomics.org/plink/1.9/formats#bed) format (see [codecs.py](codecs.py))

Notably, the 3.8G PLINK bed file used in this prototype compresses to 1.4G using Zarr (w/ bitpacking and default Blosc/LZ4 codec).