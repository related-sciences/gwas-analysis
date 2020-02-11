import os.path as osp
PLINK_FILE_TGT = 'cornell_canine'
DATA_DIR_TGT = osp.expanduser('~/data/gwas/canine-hayward-2016-266k4')
PLINK_FILE_REF = 'All_Pure_150k'
DATA_DIR_REF = osp.expanduser('~/data/gwas/canine-parker-2017')
REF_GENOME_PATH = osp.join('data', 'reference_genome.json')