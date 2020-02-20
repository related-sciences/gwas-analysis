## LSH POC

The purpose of this experiment is to evaluate how well a simple LSH scheme can approximate LD for pairs of variants.  The notebooks in this small pipeline will:

1. Sample variants from some dataset
2. Compute ground-truth LD and store only pairs of variants passing a low minimum threshold
3. Compute LD implied by LSH hash collisions
4. Evaluate precision and recall of pairwise LD recognition by hash collision as a function of:
    - The number of random projections used in the hash
    - The LD threshold defining the ground truth links

The hashing approach applied here is based largely on [Shared Nearest Neighbor Clustering in a Locality Sensitive Hashing Framework](https://www.ncbi.nlm.nih.gov/pubmed/28953425), which details a method for clustering genomic sequence data.