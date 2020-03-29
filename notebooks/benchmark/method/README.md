## Method Implementations

### LD Pruning

Two attempts have been made to implement LD pruning with different algorithms or accelerated versions of existing ones:  

- [Locality Sensitive Hashing](ld_prune/lsh/README.md) - The README for this experiment covers the premise and results well, with the conclusion being that it will take too many projection vectors to do this efficiently for low R2 thresholds (it would be effective at high thresholds though).  Experiments with orthogonalization of random projections yielded little gains.
- [Triangle Inequality](ld_prune/tild/README.md) - This README has the details, but this implementation has a numba-accelerated python implementation of the more common PLINK pruning routine.  The triangle inequality trick added to the implementation makes a huge difference at higher r2 pruning thresholds, but has little effect at more common lower thresholds (unfortunately).
