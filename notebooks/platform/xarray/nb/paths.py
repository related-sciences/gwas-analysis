# Various paths used within demo notebooks
import os
from pathlib import Path
PLINK_HAPMAP_PATH_01 = Path(os.environ['DATA_DIR']) / 'gwas/tutorial/1_QC_GWAS/HapMap_3_r3_1'
PLINK_1KG_PATH_01 = Path(os.environ['DATA_DIR']) / 'gwas/tutorial/2_PS_GWAS/ALL.2of4intersection.20100804.genotypes'
PLINK_1KG_PATH_02 = Path(os.environ['DATA_DIR']) / 'gwas/tutorial/2_PS_GWAS/1kG_MDS5'
CSV_1KG_POP = Path(os.environ['DATA_DIR']) / 'gwas/tutorial/2_PS_GWAS/20100804.ALL.panel'