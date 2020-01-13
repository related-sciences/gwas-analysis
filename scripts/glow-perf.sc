// Script to test distinct count over VCF data w/ Glow outside of Almond
import $ivy.`com.github.mjakubowski84::parquet4s-core:1.0.0`
import $ivy.`com.github.pathikrit::better-files:3.8.0`
import $ivy.`sh.almond::almond-spark:0.6.0`
import $ivy.`org.apache.spark::spark-sql:2.4.4` // Necessary with older versions of almond that support 2.11
import $ivy.`io.projectglow::glow:0.2.0`
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame
import better.files._
import File._
import java.io.{File => JFile}
Logger.getLogger("org").setLevel(Level.WARN)


val ss = {
  SparkSession
    .builder()
    .config("spark.sql.shuffle.partitions", "1")
    .config("spark.ui.enabled", "false")
    .config("spark.driver.host", "localhost")
    .master("local[*]")
    .getOrCreate()
}
import ss.implicits._

val data_dir = home / "data" / "gwas" / "tutorial" / "1_QC_GWAS"
val path = data_dir / "HapMap_3_r3_1.parquet"

val df = ss.read.format("plink").load(data_dir / "HapMap_3_r3_1.bed" toString)

def getNumSamples(df: DataFrame) = df
    .select(explode($"genotypes").as("genotypes"))
    .select($"genotypes.sampleId").dropDuplicates.count
def getNumVariants(df: DataFrame) = df
    .select($"names"(0)).dropDuplicates.count
    
def time[T](block: => T): T = {
    val start = System.currentTimeMillis
    val res = block
    val elapsed = System.currentTimeMillis - start
    println("Elapsed time: %.1f seconds".format(elapsed.toDouble / 1000.0))
    res
}

println((1 to 5).foreach(i => time {
    s"""
    Num samples = ${getNumSamples(df)} 
    Num variants = ${getNumVariants(df)}
    """
}))