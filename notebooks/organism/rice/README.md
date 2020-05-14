## Rice Study Data


- SNP-seek
    - Downloads: https://snp-seek.irri.org/_download.zul
        - "3K RG 29mio biallelic SNPs" is used here
        - Phenotypes are in "3K RG morpho-agronomic data, MS Excel format"
    - [GWAS Results Search Page](https://snp-seek.irri.org/_gwas.zul)
    - [Publications Page](https://snp-seek.irri.org/_about.zul)
        - Primary publication: Rice SNP-seek database update: new SNPs, indels, and queries. Nucl. Acids Res.(2017)
            - Mansueto, et al.
            - [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5210592/)
            - *This paper describes new features and datasets added to SNP-Seek in 2015-2017 as well as software and database updates.*
        - Update: SNP-Seek II: A resource for allele mining and analysis of big genomic data in Oryza sativa. Current Plant Biology (2016)
            - Mansueto, et al. 
            - [Semantic Scholar](https://www.semanticscholar.org/paper/SNP-Seek-II%3A-A-resource-for-allele-mining-and-of-in-Mansueto-Fuentes/1fc00cb83e927827a7855fda9912abd551132dd2)
            - *This paper contains details on variant calling for 5 references, integration of additional genomic data, web interface, database schema, use cases, web services API.*
        - Original: SNP-Seek database of SNPs derived from 3000 rice genomes. Nucl. Acids Res. (2015).
            - Alexandrov, et al.
            - [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4383887/)
            - *This is the first publication introducing SNP-Seek database, data generation for Nipponbare reference,and common use cases.*
        - Other: Genomic variation in 3,010 diverse accessions of Asian cultivated rice. Nature. (2018)
            - *The main scientific paper for the 3K RGP.*
    - SNP-Seek derived from [The 3,000 rice genomes project](https://www.ncbi.nlm.nih.gov/pubmed/24872877) (2014)
    - Data Subsets
        - These seem to have been defined in "Genomic variation in 3,010 diverse accessions of Asian cultivated rice"
        - In particular for the "core" set:
            > (3) a core SNP set of SNPs derived from the filtered SNP set using a two-step linkage disequilibrium pruning procedure with PLINK, in which SNPs were removed by linkage disequilibrium pruning with a window size of 10 kb, window step of one SNP and r2 threshold of 0.8, followed by another round of linkage disequilibrium pruning with a window size of 50 SNPs, window step of one SNP and r2 threshold of 0.8.
        - On the number of samples / snps used in GWAS:
            > Extended Data Fig. 10 
            > Genome-wide association for grain length, grain width and bacterial blight isolate C5. a–c, 
            > GWAS for grain length (GRLT, n = 2,012) (a), grain width (GRWD, n = 2,012) (b) and bacterial blight isolate C5 (BBL C5, n = 381) (c). 
            > GWAS was performed using filtered and linkage disequilibrium-pruned SNPs for historical trait data on source accessions for grain length and grain width (223,743 SNPs) and for newly acquired lesion length data for bacterial blight isolate C5 (148,999 SNPs).

