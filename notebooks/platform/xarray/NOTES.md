# Notes

Notes on prototype components, generally related to implementation details.  Higher-level concerns go to issues/discourse instead.
   
## Data Structures

- `xr.DataArray(xr.DataArray())` will call `.asarray` on input argument pushing .data into memory as numpy
- Accessing "owner" class when working with descriptors is possible with __set_name__
  - See: https://www.python.org/dev/peps/pep-0487/
  - See: https://stackoverflow.com/questions/2366713/can-a-decorator-of-an-instance-method-access-the-class
- Copying docstrings in wrapped functions: https://docs.python.org/2/library/functools.html#functools.update_wrapper

### Array Backends

- [Sparse Array Support](https://github.com/pydata/xarray/issues/1375)
    - Stephen mentioned you can find duck array hacks with `https://github.com/pydata/xarray/search?p=1&q=dask_array_type&type=&utf8=%E2%9C%93`
- [Duck Array Support](https://github.com/pydata/xarray/issues/1938)
    - Called "Hooks for XArray operations"
    - This is what led xarray to `duck_array_ops`, rather than hardcoded dask/numpy switches
    - There was a proposal for https://github.com/hameerabbasi/arrayish that has now become:
        - https://uarray.org/en/latest/
            - "uarray is a backend system for Python that allows you to separately define an API, along with backends that contain separate implementations of that API."
            - "unumpy builds on top of uarray"
        - https://unumpy.uarray.org/en/latest/
            - " It is an effort to specify the core NumPy API, and provide backends for the API"
 
### Outstanding Questions

- Is there really no compatible type hint for duck arrays?
  - Xarray uses Any (e.g. [here](https://github.com/pydata/xarray/blob/master/xarray/core/dataarray.py#L559)), dask uses no hints
  - Everybody is waiting on [PEP 484](https://github.com/numpy/numpy/issues/7370) rather than supporting temporary workarounds
- What is the best way to manage optional dependencies (i.e. avoid requiring global dask imports)?
   - see [pandas.compat.import_optional_dependency](https://github.com/pandas-dev/pandas/blob/3a5ae505bcec7541a282114a89533c01a49044c0/pandas/compat/_optional.py#L47) for how pandas manages IO backend module imports
   
## IO

### Plugin Systems

- Scikit-image
  - https://scikit-image.org/docs/dev/user_guide/plugins.html
  - https://github.com/scikit-image/scikit-image/blob/f1b7cf60fb80822849129cb76269b75b8ef18db1/skimage/io/manage_plugins.py#L173
     - use_plugin takes name for plugin and name of function to make default for (e.g. imread, imsave, etc.)
       - by default, it overrides all functions with plugin default
     - There is no external framework around it, it is more of a way to "To improve performance, plugins are only loaded as needed"
     - an ini file lets the developer know what functions are provided by plugin without importing it
  - Something like this might be better with Yapsy
  - Disadvantage: Dependencies of plugins are not an explicit part of the build process
- Pandas
  - dependent modules are checked and imported in __init__ of abstract class implementations
  - resolving an engine: https://github.com/pandas-dev/pandas/blob/master/pandas/io/parquet.py#L14
  - pandas read_html uses "flavor" argument
  - uses module = importlib.import_module(name) [here](https://github.com/pandas-dev/pandas/blob/3a5ae505bcec7541a282114a89533c01a49044c0/pandas/compat/_optional.py#L47)
    and uses module reference as something akin to a plugin descriptor
  - options like `io.parquet.engine` set globally for default
  - read_excel takes engine argument (“xlrd”, “openpyxl” or “odf”)
- Setuptools
  - This provides a way for a lib to expose entry points for devs of packages to hook into (through 'entry_points')
  - https://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins
- Flask
  - Allows for plugins to be written as PyPi packages
  - https://flask.palletsprojects.com/en/1.1.x/extensiondev
  - https://github.com/pallets/flask/blob/2062d984abd3364557a6dcbd3300cfe3e4ecf156/docs/cli.rst#plugins
    - Flask loads flask.commands entry_point specified by extension package in 'entry_points' (in setup.py)