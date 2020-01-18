# GWAS Tutorial Publication Notebooks

These notebooks are based on the GWAS analysis walk-through in [A tutorial on conducting genome‐wide association studies: Quality control and statistical analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6001694/).

### Hail Notes

To launch Hail:

```bash
export HAIL_HOME=/opt/conda/envs/hail/lib/python3.7/site-packages/hail
pyspark \
  --jars $HAIL_HOME/hail-all-spark.jar \
  --conf spark.driver.memory=32g \
  --conf spark.driver.extraClassPath=$HAIL_HOME/hail-all-spark.jar \
  --conf spark.executor.extraClassPath=./hail-all-spark.jar \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.kryo.registrator=is.hail.kryo.HailKryoRegistrator
```

Example session:

```python
import hail as hl
import os.path as osp
hl.init(sc=spark.sparkContext) 

data_dir = osp.expanduser('~/data/gwas/tutorial/1_QC_GWAS')
mt = hl.import_plink(
    osp.join(data_dir, 'HapMap_3_r3_1.bed'),
    osp.join(data_dir, 'HapMap_3_r3_1.bim'),
    osp.join(data_dir, 'HapMap_3_r3_1.fam'),
    skip_invalid_loci=True
)
mt_pr = hl.ld_prune(mt.GT, r2=0.2, bp_window_size=500000) 
print(mt_pr.entry.take(5))
```

IPython and Jupyter can be used too, but they need to be initialized with some of the properties above (driver memory at the very least).  This isn't actually documented well but instructions like in this [post](https://beehive.cs.princeton.edu/category/uncategorized/) could probably help.

**Installing BLAS**

```sudo apt-get install libatlas-base-dev```

### Glow Notes


#### VCF

There is undocumented code in Glow that makes it possible to add INFO fields to a vcf file by including DataFrame fields with the prefix "INFO_":

See: https://github.com/projectglow/glow/blob/94f837609ae240ed68c0ea83f42fd93ca144b096/core/src/main/scala/io/projectglow/common/schemas.scala#L55

Example:

```scala
ss.read.format("plink").load(data_dir / "HapMap_3_r3_1.bed" toString)
    .repartition(1)
    .withColumn("INFO_p1", lit("yes"))
    .filter(!$"referenceAllele".isin("I", "D"))
    .limit(100)
    .repartition(1).write.format("vcf").mode("overwrite").save("/tmp/df_qc_4.vcf")
```

### PLINK Notes

Writing VCFs can cause issues if the allele encodings include deletions/insertions.  To check if a plink dataset can be written w/o issues, look for these warnings:

```bash
plink --bfile HapMap_3_r3_1 --recode vcf --biallelic-only --out HapMap_3_r3_1.vcftest
# Warning: 225 het. haploid genotypes present (see HapMap_3_r3_1.vcftest.hh );
# many commands treat these as missing.
# Warning: At least one VCF allele code violates the official specification;
# other tools may not accept the file.  (Valid codes must either start with a
# '<', only contain characters in {A,C,G,T,N,a,c,g,t,n}, be an isolated '*', or
# represent a breakend.)
```

#### Data Notes

1000 Genomes data size:

```bash
# From: ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/release/20100804/ALL.2of4intersection.20100804.genotypes.vcf.gz
gzip -cd ALL.2of4intersection.20100804.genotypes.vcf.gz | wc
25488507 16261656124 501287626783 // ~500G decompressed
```

### Liftover Notes

- [chain file spec](https://genome.ucsc.edu/goldenPath/help/chain.html)
- [hail variant liftover function](https://hail.is/docs/0.2/guides/genetics.html#liftover-howto )