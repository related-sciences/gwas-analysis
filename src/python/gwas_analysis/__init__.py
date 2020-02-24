import os
import os.path as osp
from dotenv import dotenv_values

if 'NB_DIR' not in os.environ:
    raise EnvironmentError('Environment variable "NB_DIR" must be set')

# Add environment variables as globals in notebooks
for k, v in dotenv_values(osp.join(os.environ['NB_DIR'], 'env.sh')).items():
    globals()[k] = v


PKG_DIR = osp.abspath(osp.dirname(__file__))