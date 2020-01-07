## GWAS Analysis

### Setup

Hail setup test in python notebook:

import hail as hl
mt = hl.balding_nichols_model(n_populations=3, n_samples=50, n_variants=100)
mt.count()


Glow test in Almond:

```scala
import $ivy.`org.apache.spark::spark-sql:2.4.4`
import $ivy.`sh.almond::almond-spark:0.6.0`
import $ivy.`io.projectglow::glow:0.2.0`
import io.projectglow.Glow
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
val ss = {
  NotebookSparkSession
    .builder()
    .config("spark.sql.shuffle.partitions", "1")
    .config("spark.ui.enabled", "false")
    .config("spark.driver.host", "localhost")
    .master("local[*]")
    .getOrCreate()
}
import ss.implicits._
Glow.register(ss)
df = ss.read.format("plink").load("/home/eczech/data/1_QC_GWAS/HapMap_3_r3_1.bed")
```
