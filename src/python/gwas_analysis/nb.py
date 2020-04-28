# Alternative that works for both Python 2 and 3:
from requests.compat import urljoin

import gwas_analysis
import ipykernel
import json
import os
import os.path as osp
import re
import requests


#############
# NB Identity
#############
# See: https://github.com/jupyter/notebook/issues/1000


try:  # Python 3 (see Edit2 below for why this may not work in Python 2)
    from notebook.notebookapp import list_running_servers
except ImportError:  # Python 2
    import warnings
    from IPython.utils.shimmodule import ShimWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ShimWarning)
        from IPython.html.notebookapp import list_running_servers


def get_notebook_path():
    """
    Return the full path of the jupyter notebook.
    """
    kernel_id = re.search(
        "kernel-(.*).json", ipykernel.connect.get_connection_file()
    ).group(1)
    servers = list_running_servers()
    for ss in servers:
        response = requests.get(
            urljoin(ss["url"], "api/sessions"), params={"token": ss.get("token", "")}
        )
        for nn in json.loads(response.text):
            if nn["kernel"]["id"] == kernel_id:
                relative_path = nn["notebook"]["path"]
                return osp.join(ss["notebook_dir"], relative_path)


def get_notebook_relpath(start=gwas_analysis.NB_DIR):
    return osp.relpath(get_notebook_path(), start)


def get_notebook_name():
    return osp.splitext(osp.basename(get_notebook_path()))[0]


def get_temp_file(path, filename, sep="-"):
    nb_name = get_notebook_name()
    path = osp.join(path, nb_name)
    if not osp.exists(path):
        os.makedirs(path)
    return osp.join(path, filename)
