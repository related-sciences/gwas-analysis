
/*
 * Time execution of an operation
 */
def time[T](block: => T): T = {
    val start = System.currentTimeMillis
    val res = block
    val elapsed = System.currentTimeMillis - start
    println("Elapsed time: %.1f seconds".format(elapsed.toDouble / 1000.0))
    res
}