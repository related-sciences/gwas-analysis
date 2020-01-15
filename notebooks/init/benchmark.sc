import $file.^.init.paths, paths._
import better.files.File

/*
 * Time execution of an operation
 */
def timer[T](key: String)(op: String)(block: => T): T = {
    val start = System.currentTimeMillis
    val res = block
    val curr = System.currentTimeMillis
    val elapsed = curr - start
    File("benchmarks.csv").appendLine(s"$curr\t$key\t$op\t$elapsed")
    println("Elapsed time: %.1f seconds".format(elapsed.toDouble / 1000.0))
    res
}