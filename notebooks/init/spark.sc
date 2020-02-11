import $ivy.`sh.almond::almond-spark:0.6.0`
import $ivy.`org.apache.spark::spark-sql:2.4.4` // Necessary with older versions of almond that support 2.11
import org.apache.spark.sql._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame
Logger.getLogger("org").setLevel(Level.WARN)

// Spark session generator for local mode analyses 
def getLocalSparkSession(
    enableProgress: Boolean = false, 
    keepProgress: Boolean = false, 
    enableUI: Boolean = false, 
    uiPort: Int = 4040,
    shufflePartitions: Int = 200,
    master: String = "local[*]",
    driverHost: String = "localhost",
    broadcastTimeoutSeconds: Int = 300 
  ) = {
  NotebookSparkSession
    .builder()
    // See https://github.com/almond-sh/almond/blob/620011b6edd152a84d3ac2637d45620a8b95af02/modules/scala/almond-spark/src/main/scala/org/apache/spark/sql/almondinternals/NotebookSparkSessionBuilder.scala
    .progress(enable=enableProgress, keep=keepProgress)
    // Default is 200; see https://spark.apache.org/docs/latest/sql-performance-tuning.html
    .config("spark.sql.shuffle.partitions", shufflePartitions)
    // Default is 300; see https://spark.apache.org/docs/latest/sql-performance-tuning.html
    .config("spark.sql.broadcastTimeout", broadcastTimeoutSeconds)
    .config("spark.ui.enabled", enableUI)
    .config("spark.ui.port", uiPort)
    .config("spark.driver.host", driverHost)
    .master(master)
    .getOrCreate()
}

implicit class DFOPs(df: DataFrame) {
    def fn[T](fn: DataFrame => T): T = fn(df)
}