# Share environment variables and path-related functions
# DO NOT move this file, it is the only file for which relative paths must remain fixed (for notebooks)

import os.path as osp
from dotenv import dotenv_values, find_dotenv
import gwas_analysis.benchmark as gab

# Add environment variables as globals in notebooks
for k, v in dotenv_values(find_dotenv('env.sh')).items():
    globals()[k] = v

def plink_files(dir_name, file_name):
    return [osp.join(dir_name, file_name + ext) for ext in ['.bed', '.bim', '.fam']]