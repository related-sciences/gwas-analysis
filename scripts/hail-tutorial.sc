# Example hail commands (need to configure jupyter and move to notebook)
import hail as hl
import os.path as osp
hl.init(sc=spark.sparkContext) 

data_dir = osp.expanduser('~/data/gwas/tutorial/1_QC_GWAS')
mt = hl.import_plink(
    osp.join(data_dir, 'HapMap_3_r3_1.bed'),
    osp.join(data_dir, 'HapMap_3_r3_1.bim'),
    osp.join(data_dir, 'HapMap_3_r3_1.fam'),
    skip_invalid_loci=True
)
mt.write(osp.join(data_dir, 'HapMap_3_r3_1.mt'), overwrite=True)

mt = hl.read_matrix_table(osp.join(data_dir, 'HapMap_3_r3_1.mt'))

mt = mt.annotate_cols(pheno = table[mt.s])

# Check .fam data
mt.col.describe()

# Add sample and variant qc
mt = hl.sample_qc(mt)
mt = hl.variant_qc(mt)
mt.col.describe()

pd.Series( mt.sample_qc.call_rate.collect() ).describe()

pd.Series( mt.variant_qc.call_rate.collect() ).describe()

imputed_sex = hl.impute_sex(mt.GT)
mt_fs = mt.filter_cols(imputed_sex[dataset.s].is_female != dataset.pheno.is_female, keep=False)

mt_pr = hl.ld_prune(mt.GT, r2=0.2, bp_window_size=500000)
print(mt_pr.entry.take(5))