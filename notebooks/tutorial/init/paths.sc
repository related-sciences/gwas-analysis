import $ivy.`com.github.pathikrit::better-files:3.8.0`
import better.files._
import File._
import java.io.{File => JFile}
val GWAS_TUTORIAL_DATA_DIR = home / "data" / "gwas" / "tutorial"

val QC0_FILE = "HapMap_3_r3_1"
val QC1_FILE = "HapMap_3_r3_5"
val QC2_FILE = "HapMap_3_r3_6"
val QC3_FILE = "HapMap_3_r3_8"
val QC4_FILE = "HapMap_3_r3_9"
val QC5_FILE = "HapMap_3_r3_10"