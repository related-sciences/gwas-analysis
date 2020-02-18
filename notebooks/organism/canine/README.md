## Canine GWAS 


#### Overview

This analysis is intended to emulate the QC procedure used by UK Biobank in QC [Bycroft et al. 2018](https://www.nature.com/articles/s41586-018-0579-z#MOESM1) for a large cohort with a smaller cohort of known ancestry (and no recorded phenotypes), i.e. 1000 Genomes.  In this case, we will use genotyping data from the NHGRI Dog Genome Project for 1,355 dogs of known breed (the 1KG analog) as a means to better understand and prepare data from a larger dataset (4,3442 dogs) shared in [Hayward et al. 2016](https://www.nature.com/articles/ncomms10460) (the UKBB analog).  The former will be refered to as the "reference" dataset and the latter as the "target" dataset throughout this doc.

#### Outline

Below is an outline of the UKBB QC and analysis process (taken from this [supplement](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0579-z/MediaObjects/41586_2018_579_MOESM1_ESM.pdf) to Bycroft et al. 2018 and referred to as "UKBS" hereafter) and a mapping to similar steps we can take with the canine data:

- Variant QC
    - UKBB Variant QC
        - The general flow for the UKBB variant QC process is:
            1. Use 1000 Genomes (1KG) data to create a PC space onto which UKBB samples can be projected
            2. Select only **samples** near the 1KG European samples 
            3. Use the selected samples to do any further variant QC (ensuring that MAF and HWE tests are based on homogenous samples is crucial)
            4. Feed the passing variants to the later Sample QC step
        - More specifically, here are details on some of these steps:
            - 1000 Genomes (1KG) QC (UKBS S2.2):
                > We first downloaded 1000 Genomes Project Phase 1 data in Variant Call File (VCF) format and extracted 714,168 SNPs (no Indels) that are also on the UK Biobank Axiom array.  We selected 355 unrelated samples from the populations CEU, CHB, JPT, YRI, and then chose SNPs for principal component analysis using the following criteria:
                > - MAF ≥ 5% and HWE p-value > 10-6 in each of the populations CEU, CHB, JPT and YRI.  
                > - Pairwise r2 ≤ 0.1 to exclude SNPs in high LD.  (The r2 coefficient was computed using plink 9 and its indep-pairwise function with a moving window of size of 1000 kilo-bases (Kb) and a step-size of 80 markers).
                > - Removed C/G and A/T SNPs to avoid unresolvable strand mismatches.
                > - Excluded SNPs in several regions with high PCA loadings (after an initial PCA).
            - PCA projection
                - UKBS S 2.2:
                    > With the remaining 40,220 SNPs we computed PCA loadings from the 355 1,000 Genomes samples, then projected all the UK Biobank samples onto the 1st and 2nd principal components
            - Sample selection
                - UKBS S 2.2:
                    > Many QC tests are ineffective in the context of population structure.  We therefore applied all marker-based QC tests using a subset of 463,844 individuals drawn from the largest ancestral group in the cohort (European).  Here we describe the procedure to identify such samples using principal component analysis (PCA) and two-dimensional clustering.  
                - The UKBB process identifies samples in the projected PCA space using a clustering algorithm called *aberrant*.  They do this primarily because samples need to be selected for each batch.  The whole experiment includes "106 batches of around 4,700 samples each (~4,800 including the controls)". 
            - Variant (i.e. Marker) selection
                - UKBS S 2.3:
                    > Specifically, we tested for batch effects, plate effects, departures from Hardy-Weinberg equilibrium (HWE), sex effects, array effects, and discordance across control replicates.  All tests (except for discordance across controls) were applied using the genotype calls for a set of 463,844 ancestrally homogeneous individuals (see Section S 2.2). 
    - Canine Variant QC
        - Our adpated version of this QC process will look like this:
            - Variant QC 
                - Group the NHGRI Dog Genome (i.e. "reference") samples by the 3 most common breeds
                - Run MAF and HWE filtering on all variants for each breed separately, and identify variants that pass all filters in all breeds
                - Filter the original reference data to the filtered variant set
                - Run LD pruning
                - Remove C/G and A/T SNPs
                - Exclude SNPs with high PCA loadings
                    - This appears to mean that after running PCA, peaks are identified in loadings and then removed manually before running another round of PCA
                - Align the (now fairly small) set of reference variants to the target variants
                    - This step is missing in the UKBB analysis because presumably, the 1KG data and the UKBB data are already aligned to GRCh38
                    - This will involve joining variants by locus and resolving strand/allele flips
                    - This will result in a merged target + reference dataset that contains **exactly** the same variants (it is crucial that the reference data PCA include only variants that are going to be present in the target dataset to be projected)
                - Run PCA on the merged dataset filtered to reference samples
                - Investigate high loadings
                - Potentially re-run PCA if SNPs need to be removed
                    - i.e. take a guess at what this means "Excluded SNPs in several regions with high PCA loadings (after an initial PCA)."
                - Project all target samples onto the reference PCA space
                    - The 1KG data is necessary to create PCs from a well QC'd sample (and then project the target data onto it)
                    - You could just use PCA to cluster the original dataset, but you would not know which samples are from which superpopulation
                - Select target samples in the neighborhood of a homogeneous labeled population (Boxers in the reference data)
                - Run HWE filtering on all target variants in the selected sample
                - Subset the original target dataset to only variants passing all filters
                - **Output**: A version of the original target dataset with fewer variants
    -  Canine Sample QC
        - See [Sams and Boyko 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6325901/) for a reference on using very high IBD threshold of .45 to eliminate related pairs
            
#### Canine Datasets

- Reference Dataset
    - The NHGRI Dog Genome Project
    - In [genome summary](https://research.nhgri.nih.gov/dog_genome/canine_genome.shtml) page, this is cited as original dog genome publication: [Genome sequence, comparative analysis and haplotype structure of the domestic dog](https://www.ncbi.nlm.nih.gov/pubmed/16341006)
        - A single female boxer was used to generate draft genome sequence
        - A table of contig sizes is available in the [supplementary notes](https://static-content.springer.com/esm/art%3A10.1038%2Fnature04338/MediaObjects/41586_2005_BFnature04338_MOESM2_ESM.pdf) (Section B: CanFam2.0)
    - https://research.nhgri.nih.gov/dog_genome/data_release/index.shtml
    - [Genomic Analyses Reveal the Influence of Geographic Origin, Migration, and Hybridization on Modern Dog Breed Development](https://www.sciencedirect.com/science/article/pii/S2211124717304564?via%3Dihub)
        - Parker et al. 2017 (last author Elaine Ostrander)
        - [FTP site](ftp://ftp.nhgri.nih.gov/pub/outgoing/dog_genome/SNP/)
        - This includes a dataset with 150k snps for 1.3k dogs (as PLINK)
        - Includes dog breed
        - [Spreadsheet](https://ars.els-cdn.com/content/image/1-s2.0-S2211124717304564-mmc2.xlsx) containing breed abbreviation mappings
        - On data size: 
         > We examined genomic data from the largest and most diverse group of breeds studied to date, amassing a dataset of 1,346 dogs representing 161 breeds
        - Data was combined from several studies including the target study to be compared to below, however only ~400 samples of ~4.2k were used (not sure why more weren't used yet)
        > Samples from 938 dogs representing 127 breeds and nine wild canids were genotyped using the Illumina CanineHD bead array following standard protocols. Data were combined with publically available information from 405 dogs genotyped using the same chip (Hayward et al., 2016, Vaysse et al., 2011).
    - [Whole genome sequencing of canids reveals genomic regions under selection and variants influencing morphology](https://www.nature.com/articles/s41467-019-09373-w)
        - Plassais et al. 2019
        - [FTP site](ftp://ftp.nhgri.nih.gov/pub/outgoing/dog_genome/WGS/)
        - This WGS dataset includes a 309GB vcf, but is not used here 
- Target Dataset
    - https://datadryad.org/stash/dataset/doi:10.5061/dryad.266k4
    - [Complex disease and phenotype mapping in the domestic dog](https://www.nature.com/articles/ncomms10460)
        - Hayward et al. 2016 (last author Boyko)
        - Sample size:
        > Here we undertake the largest canine genome-wide association study to date, with a panel of over 4,200 dogs genotyped at 180,000 markers
        - Calls:
            - PLINK data sets were generated in GenomeStudio using the PLINK report plugin. 
            - Genotypes were called using a GenCall threshold of 0.15 using cluster positions that were computed for the first 30 plates
        - SNP QC:
            - In PLINK v1.07 (ref. 51), SNPs with a genotyping rate below 95% were removed. 
            - Duplicate samples were merged and discordant SNPs between the duplicates were identified and removed. 
            - SNPs with a MAF over 2% were tested for unexpected deviations from Hardy–Weinberg equilibrium. Specifically, SNPs with heterozygosity ratios (observed versus expected number of heterozygotes under Hardy–Weinberg equilibrium) below 0.25 or above 1.0 were identified and removed. 
            - Furthermore, all Y chromosome and mitochondrial DNA SNPs with any heterozygous genotype calls were removed. 
            - In total, 180,117 SNPs remained after filtering, with an overall call rate of >99.8%. The concordance rate between 44 technical replicates was 99.99%.
        - Sample QC:
            - Samples with >10% missing genotypes or with recorded sex not matching genotypic sex were excluded from further analysis. 
            - Genotypic sex was computed by calculating (1) the proportion of missing Y chromosome genotypes (<50% in males, >50% in females) and (2) the homozygosity across non-PAR X chromosome markers using the PLINK --check-sex option (generally <60% in females, >60% in males). In this manner, XXY samples were not misidentified as females and females with highly inbred X chromosomes were not misidentified as males. 
            - To check the recorded breed of our samples, we used the PLINK --genome option to check that each individual is most closely related to other individuals of the same breed
            - we also ran a principal component analysis (PCA) on each breed using the program EIGENSTRAT in the EIGENSOFT v5.0.1 package52 to identify any outliers. Dogs with recorded breed not matching the genotypic breed were excluded from further analysis.
        - On phenotypes:
            - Body size 
            > Using an additive linear model where we corrected for both inbreeding and sex of the dog (see Methods), we confirm that dog body size has a simple underlying genetic architecture3,13, with the identified 17 QTLs explaining 80–88% of the variation of weight and height in individual purebred dogs
            
#### Plotly

To restart the Orca server if it dies:

```
import plotly.io as pio
pio.orca.shutdown_server() # Then rerun plots
```

To add a vertical line annotation:

```
x_intercept = .5
fig.add_shape(dict(
    type="line", xref="x", yref="paper",
    x0=x_intercept, y0=0, x1=x_intercept, y1=1
))
```