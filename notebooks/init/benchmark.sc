import $file.^.init.paths, paths._
import better.files.File

val SESSION = System.currentTimeMillis
val DEFAULT_PATH_FORMAT = "benchmarks/benchmarks_%s_%s.csv"

/*
 * Time execution of an operation
 */
def optimer[T](key: String, op: String, block: => T, path_format: String = DEFAULT_PATH_FORMAT): T = {
    val start = System.currentTimeMillis
    val res = block
    val curr = System.currentTimeMillis
    val elapsed = curr - start
    File(path_format.format(key, SESSION)).appendLine(s"$SESSION\t$curr\t$key\t$op\t$elapsed")
    println("Elapsed time: %.1f seconds".format(elapsed.toDouble / 1000.0))
    res
}