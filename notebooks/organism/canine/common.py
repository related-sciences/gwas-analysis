import pandas as pd 
import os.path as osp
import json

def plink_files(dir_name, file_name):
    return [osp.join(dir_name, file_name + ext) for ext in ['.bed', '.bim', '.fam']]

def load_reference_genome(path):
    with open(path, 'r') as fd:
        return json.load(fd)
    
def get_bim(data_dir, data_file):
    return pd.read_csv(
        osp.join(data_dir, data_file + '.bim'), sep='\s+', header=None, 
        names=['contig', 'snp', 'pos', 'locus', 'alt', 'ref'])

def get_fam(data_dir, data_file):
    return pd.read_csv(
        osp.join(data_dir, data_file + '.fam'), sep='\s+', header=None, 
        names=['fid', 'iid', 'iid_paternal', 'iid_maternal', 'sex', 'pheno'])