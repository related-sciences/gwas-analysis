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
            - Group the NHGRI Dog Genome (i.e. "reference") samples by the 3 most common breeds
            - Run MAF and HWE filtering on all variants for each breed separately, and identify variants that pass all filters in all breeds
            - Filter the original reference data to the filtered variant set
            - Run LD pruning
            - Remove C/G and A/T SNPs
            - Align the (now fairly small) set of reference variants to the target variants
                - This step is missing in the UKBB analysis because presumably, the 1KG data and the UKBB data are already aligned to GRCh38
                - This will involve joining variants by locus and resolving strand/allele flips
                - This will result in a merged target + reference dataset that contains **exactly** the same variants (it is crucial that the reference data PCA include only variants that are going to be present in the target dataset to be projected)
            - Run PCA on the merged dataset filtered to reference samples
            - Project all target samples onto the reference PCA space
            - Select target samples in the neighborhood of homogeneous labeled populations (in the reference set)
            - For each of the target sample groupings, run HWE and MAF filtering on all target variants
            - Identify variants that pass filters in all populations
            - Subset the original target dataset to only variants passing all filters
            - **Output**: A version of the original target dataset with fewer variants
            
            
#### Notes

- Hail
    - Projecting samples using pre-computed PCs
        - Hail does not have this but McArthur lab has examples (see [discourse post](https://discuss.hail.is/t/pca-to-output-allele-frequencies-alongside-loadings/439/6) and linked [code](https://github.com/macarthur-lab/gnomad_hail/blob/537cb9dd19c4a854a9ec7f29e552129081598399/utils/generic.py#L105))
- C/G and A/T SNPs
    - This QC step is common because the probe sequence on genotyping arrays is not specific to a reference genome, by design.  This means that the genotyping data can tell you that an "A" nucleotide was present at a locus but it doesn't actually know if this nucleotide represents some kind of "variant" with respect to a larger population.  The probes are chosen such that they capture individual sites of common variation but deciding which nucleotides comprise heterozygous, homozygous ref, homozygous alt genotypes (i.e. make a call) is up to the user.  For any given site, the arrays capture multiple individual nucleotides so one way to do this independent of a reference genome, for a single dataset, is to simply assume that whatever nucleotide is less common is the minor (aka alternate) allele and the other is the major (aka reference) allele.  This is an acceptable (and very common) method for analyzing a single dataset but causes obvious problems when trying to compare calls for the same variants between datasets (a nucleotide may have been the alternate in one and reference in the other).  Two strategies for making datasets comparable then are:  
        1. For each dataset, use knowledge of the probe sequences to determine what strand each nucleotide is on.  This appears to be the only completely unambiguous way to ensure that all calls correspond to the same reference and alternate nucleotides.
        2. Use the fact that the SNP arrays at least tell you which nucleotides were measured for each locus to infer, in a fairly quick and dirty way, which strand the probes measured in each dataset.  Here are some examples to make this more clear.  Let Dataset 1 = D1 and Dataset 2 = D2 in each of these and assume each determine major/minor aka ref/alt alleles based on frequency (where "AC" implies A was designated as the major allele and C as the minor):
            - **Example 1 (AC vs CA)**: D1 says a variant has A as the major allele and C as the minor allele.  D2 says C is major and A is minor
                - Correction: For all calls in D2 for this variant, switch the homozygous/heterozygous interpretation (presumably the C allele was more common in D2 but not D1)
            - **Example 2 (AC vs TG)**: D1 says variant has A = major, C = minor and D2 says T = major, G = minor
                - Correction: Nothing for the calls.  The probes in this case were for different strands but ultimately captured the same nucleotides (since A is complementary with T and C with G) AND assumed the same major/minor relationship.  The allele nucleotides in D2 should be complemented so that the variant is known as AC, but that's it.
            - **Example 3 (AC vs GT)**
                - Correction: As a combination of example 1 and 2, the call interpretation and allele nucleotides should both be swapped in D2 to align with D1.
            - **Example 4 (AT vs TA)**
                - This is where things get tricky.  In example 2, we knew that the probes measured different strands simply because A or C nucleotides would be on one strand while T and G nucleotides would be on the other.  This definitive knowledge of a strand swap is key.  In the AT vs TA case, it could be that the probe measured different strands or it could be that the same strand was used but alleles occurred at different frequencies in both datasets.  We could now say something like, "If the A allele has a frequency of 5% in D1 and it has a frequency of ~5% in D2, then we can safely assume that the same strand was used for the probe".  This, however, becomes problematic as the allele frequencies 50%.  The same is true for cases like CG vs GC or even AT vs AT -- you simply can't tell which strand the probes corresponded too without knowledge of the probe seqeuences.  These sequences could be compared between the two datasets to determine if they were for the same strand, but they appear to be difficult to come by.  This is the main reason why many analyses simply through out A/T and C/G SNPs.
    - Here are some helpful discussions/papers on why this step is necessary (and on strand ambiguity in general):
        - [StrandScript: evaluation of Illumina genotyping array design and strand correction](https://www.ncbi.nlm.nih.gov/pubmed/28402386)
            > Additionally, the strand issue can be resolved by comparing the alleles to a reference genome. Yet, when
two alleles of the SNPs are complementary (A/T or C/G), the true strand remains undetermined. The only absolute solution to determine the strand is to compare the probe sequences to a reference genome, providing the probe sequences is correct
        - [Genotype harmonizer: automatic strand alignment and format conversion for genotype data integration](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4307387/)
            > However, there are two major challenges to be resolved: ... 2) the ambiguous A/T and G/C single nucleotide polymorphisms (SNPs) for which the strand is not obvious. For many statistical analyses, such as meta-analyses of GWAS and genotype imputation, it is vital that the datasets to be used are aligned to the same genomic strand.
        - [Is ‘forward’ the same as ‘plus’?… and other adventures in SNP allele nomenclature](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6099125/)

#### Canine Datasets

- Reference Dataset
    - The NHGRI Dog Genome Project
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