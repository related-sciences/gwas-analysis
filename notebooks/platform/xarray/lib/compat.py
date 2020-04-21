import importlib
import distutils
from dataclasses import dataclass
from typing import Optional


@dataclass
class PackageStatus:
    name: str
    version: Optional[str]
    installed: bool
    compatible: bool


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
