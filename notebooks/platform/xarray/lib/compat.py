import importlib
import distutils
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.pycompat import dask_array_type, sparse_array_type


def try_import(fn: callable) -> bool:
    try:
        fn()
        return True
    except ImportError:
        return False


numpy_array_type = (np.ndarray,)
xarray_array_type = (xr.DataArray,)
xarray_dataset_type = (xr.Dataset,)
pandas_dataframe_type = (pd.DataFrame,)

def import_cuda():
    # pylint: disable=import-error
    import numba.cuda.cudadrv.devicearray.DeviceNDArray


cuda_installed = try_import(import_cuda)

if cuda_installed:
    import numba.cuda.cudadrv.devicearray as ca
    cuda_array_type = (ca.DeviceNDArray,)
else:
    cuda_array_type = ()


def import_dask_dataframe():
    import dask.dataframe

dask_dataframe_installed = try_import(import_dask_dataframe)

if dask_dataframe_installed:
    import dask.dataframe as dd
    dask_dataframe_type = (dd.DataFrame,)
else:
    dask_dataframe_type = ()


@dataclass
class Requirement:
    package_name: str
    package_minimum_version: Optional[str] = None


@dataclass
class PackageStatus:
    name: str
    installed: bool
    compatible: bool
    version: Optional[str] = None


def check_package(name, minimum_version=None):
    """Check for optional dependency

    Parameters
    ----------
    name : str
        Name of package to check
    minimum_version : str
        Version string for minimal requirement.  If None, it is assumed
        that any installed version is compatible

    Returns
    -------
    PackageStatus
    """
    try:
        module = importlib.import_module(name)
    except ImportError:
        return PackageStatus(name=name, version=None, installed=False, compatible=False)

    # Check version against minimal requirement, if requested
    version, compatible = None, True
    if minimum_version:
        version = getattr(module, "__version__", None)
        if version is None:
            raise ImportError(f'Could not determine version for package {name}')
        # See: https://github.com/pandas-dev/pandas/blob/3a5ae505bcec7541a282114a89533c01a49044c0/pandas/compat/_optional.py#L99
        compatible = not distutils.version.LooseVersion(version) < minimum_version

    return PackageStatus(name=name, version=version, installed=True, compatible=compatible)
