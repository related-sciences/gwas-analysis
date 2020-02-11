import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import java.nio.file.Paths

class MatrixTable(val rows: DataFrame, val cols: DataFrame, val entries: DataFrame) {
    
    def save(path: String) = {
        if (!Paths.get(path).toFile.exists)
            Paths.get(path).toFile.mkdirs
        rows.write.format("parquet").mode("overwrite").save(Paths.get(path, "rows.parquet").toString)
        cols.write.format("parquet").mode("overwrite").save(Paths.get(path, "cols.parquet").toString)
        entries.repartition(16).write.format("parquet").mode("overwrite").save(Paths.get(path, "entries.parquet").toString)
    }
    
}

object MatrixTable {
    
    def load(path: String)(implicit ss: SparkSession) = {
        new MatrixTable(
            rows=ss.read.parquet(Paths.get(path, "rows.parquet").toString),
            cols=ss.read.parquet(Paths.get(path, "cols.parquet").toString),
            entries=ss.read.parquet(Paths.get(path, "entries.parquet").toString)
        )
    }
    
    def fromPLINKDataset(df: DataFrame, dp: DataFrame)(implicit ss: SparkSession) = {
        if (!df.schema.names.contains("variantId"))
            throw new IllegalArgumentException("Genotype data frame must contain 'variantId' field")
        if (!dp.schema.names.contains("sampleId"))
            throw new IllegalArgumentException("Pedigree data frame must contain 'sampleId' field")
        if (df.select("variantId").distinct.count != df.count)
            throw new IllegalArgumentException("Genotype field 'variantId' must be unique")
        if (dp.select("sampleId").distinct.count != dp.count)
            throw new IllegalArgumentException("Pedigree field 'sampleId' must be unique")
        
        val cols = df
            .withColumn("genotypes", explode(col("genotypes")))
            .select("genotypes.sampleId")
            .dropDuplicates("sampleId")
            .join(dp, Seq("sampleId"), "left")
            .withColumn("colId", monotonically_increasing_id())
            .withColumn("colId", row_number.over(Window.orderBy(col("colId"))))

        val rows = df
            .drop("genotypes")
            .withColumn("rowId", monotonically_increasing_id())
            .withColumn("rowId", row_number.over(Window.orderBy(col("rowId"))))


        val entries = df
            .withColumn("state", expr("genotype_states(genotypes)"))
            .withColumn("sampleId", col("genotypes.sampleId"))
            .selectExpr("variantId", "explode(arrays_zip(sampleId, state)) as gt")
            .select("variantId", "gt.*")
            .join(rows.select("rowId", "variantId"), Seq("variantId"), "inner")
            .join(cols.select("colId", "sampleId"), Seq("sampleId"), "inner")
            .select(col("rowId"), col("colId"), col("state").cast("byte").as("state"))
            .filter(col("state") =!= 0) // Ignore homozygous reference
            .sortWithinPartitions("rowId", "colId") // Sort for better compression
        
        new MatrixTable(rows=rows, cols=cols, entries=entries)
    }
}

implicit class MatrixTableOps(mt: MatrixTable) {
    
    lazy val nCols: Long = mt.cols.count
    
    lazy val nRows: Long = mt.rows.count
    
    def count = (nRows, nCols)

    def filterCols(fn: DataFrame => DataFrame): MatrixTable = {
        val cols = fn(mt.cols)
        new MatrixTable(
            rows=mt.rows,
            cols=cols,
            entries = mt.entries.join(broadcast(cols.select("colId")), Seq("colId"), "leftsemi")
        )
    }
    
    def filterRows(fn: DataFrame => DataFrame): MatrixTable = {
        val rows = fn(mt.rows)
        new MatrixTable(
            rows=rows,
            cols=mt.cols,
            entries = mt.entries.join(broadcast(rows.select("rowId")), Seq("rowId"), "leftsemi")
        )
    }
    
    def getSampleStats() = {
        mt.entries
            .groupBy("colId").agg(
                sum(when(col("state") === -1, 1).otherwise(0)).as("nUncalled"),
                sum(when(col("state") === 1, 1).otherwise(0)).as("nHet")
            )
            .join(mt.cols.select("colId"), Seq("colId"), "right")
            .withColumn("nUncalled", coalesce(col("nUncalled"), lit(0)))
            .withColumn("nCalled", lit(nRows) - col("nUncalled"))
            .withColumn("callRate", col("nCalled") / (col("nCalled") + col("nUncalled")))
    }
    
    def getVariantStats() = {
        mt.entries
            .groupBy("rowId").agg(
                sum(when(col("state") === -1, 1).otherwise(0)).as("nUncalled"),
                sum(when(col("state") === 1, 1).otherwise(0)).as("nHet")
            )
            .join(mt.rows.select("rowId"), Seq("rowId"), "right")
            .withColumn("nUncalled", coalesce(col("nUncalled"), lit(0)))
            .withColumn("nCalled", lit(nCols) - col("nUncalled"))
            .withColumn("callRate", col("nCalled") / (col("nCalled") + col("nUncalled")))
    }
    
    def transform(fn: MatrixTable => MatrixTable): MatrixTable = {
        fn(mt)
    }
}

object QCOP {

    def filterByVariantCallRate(threshold: Double)(mt: MatrixTable): MatrixTable = { 
        mt.filterRows(rows => {
            rows.join(
                mt.getVariantStats().filter(col("callRate") >= threshold).select("rowId"),
                Seq("rowId"), "leftsemi"
            )
        })
    }
    def filterBySampleCallRate(threshold: Double)(mt: MatrixTable): MatrixTable = { 
        mt.filterCols(cols => {
            cols.join(
                mt.getSampleStats().filter(col("callRate") >= threshold).select("colId"),
                Seq("colId"), "leftsemi"
            )
        })
    }

}