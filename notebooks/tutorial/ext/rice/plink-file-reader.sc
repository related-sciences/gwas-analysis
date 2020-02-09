import $file.plink, plink._
import org.apache.spark.sql.Row

//val iter = plink.PlinkReader.getRowIterator("/home/eczech/data/gwas/tutorial/2_PS_GWAS/HapMap_3_r3_13.bed")
val iter = plink.PlinkReader.getRowIterator("/home/eczech/data/gwas/rice-snpseek/3K_RG_29mio_biallelic_SNPs_Dataset/NB_final_snp.bed")

def getCallRate(r: Row) = {
    val rsid = r.getAs[Seq[String]]("names")(0)
    val gt = r.getAs[Seq[Row]]("genotypes")
    val nt = gt.size
    val na = gt.map(c => if (c.getAs[Seq[Int]]("calls").filter(_ < 0).isEmpty) 0 else 1).sum
    val cr = (nt - na).toFloat / nt.toFloat
    (rsid, cr)
}

val callRates = iter.map(getCallRate).toList
print(f"Call rates size = ${callRates.size}")