import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, explode, size}

/**
 * Count variants and samples in a Glow DataFrame
 */
def gdCount(df: DataFrame) = {
    // (n_variants, n_samples)
    (df.count, df.select(size(col("genotypes"))).first.getAs[Int](0))
}

/**
 * Count unique variants and samples in a Glow DataFrame
 */
def gdUniqueCount(df: DataFrame) = {
    val nVariants = df.select(col("names")(0)).dropDuplicates.count
    val nSamples = df
        .select(explode(col("genotypes")).as("genotypes"))
        .select(col("genotypes.sampleId")).dropDuplicates.count
    (nVariants, nSamples)
}


/**
 * Filter a genotype dataset to samples with a minimum call rate (across variants)
 */
def gdFilterBySampleCallRate(threshold: Double)(df: DataFrame): DataFrame = { 
    df
        // Cross join original dataset with single-row data frame containing a map like (sampleId -> QC stats)
        .crossJoin(
            df
            .selectExpr("sample_call_summary_stats(genotypes, referenceAllele, alternateAlleles) as qc")
            .selectExpr("map_from_arrays(qc.sampleId, qc) as qc")
        )
        // For each row, filter the genotypes array (which has one element per sampleId) based on QC map lookup
        .selectExpr("*", s"filter(genotypes, g -> qc[g.sampleId].callRate >= ${threshold}) as filtered_genotypes")
        // Remove intermediate fields 
        .drop("qc", "genotypes").withColumnRenamed("filtered_genotypes", "genotypes")
        // Ensure that the original dataset schema was preserved
        .transform(d => {assert(d.schema.equals(df.schema)); d})
}

/**
 * Filter a genotype dataset to variants with a minimum call rate (across samples)
 */
def gdFilterByVariantCallRate(threshold: Double)(df: DataFrame): DataFrame = { 
    df
        .selectExpr("*", "call_summary_stats(genotypes) as qc")
        .filter(col("qc.callRate") >= threshold)
        .drop("qc")
        .transform(d => {assert(d.schema.equals(df.schema)); d})
}