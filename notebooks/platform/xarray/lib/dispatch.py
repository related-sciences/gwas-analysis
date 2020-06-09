from __future__ import annotations
import functools
from typing import Hashable, Callable, Union, Sequence, Dict, Type, Mapping, Optional
from typing_extensions import Protocol
from .config import Configuration
from .config import config as global_config
from .compat import Requirement, check_package


# ----------------------------------------------------------------------
# Data Models

class Domain(tuple):
    """Path-like dispatching domain name model class

    To operate with a global configuration namespace and to avoid function name conflicts between distinct
    APIs, a "Domain" is used to represent the demarcation between them.  This is useful for unifying
    configuration properties with frontend and backend implementations which should all be separated
    by design, and require only the domain to indicate relationships between them.
    """
    sep = '.'

    def __new__(cls, value: Union[str, Sequence[str]]):
        if isinstance(value, str):
            value = value.split(Domain.sep)
        return super().__new__(cls, tuple(value))

    def append(self, other: Union[str, Sequence[str]]) -> Domain:
        """Append to current domain path"""
        return Domain(self + Domain(other))

    def startswith(self, other: Sequence[str]):
        """Identify shared prefix in domain path"""
        if len(self) < len(other):
            return False
        return all([p == self[i] for i, p in enumerate(other)])

    def __str__(self):
        return Domain.sep.join(self)


# ----------------------------------------------------------------------
# Protocols

class Dispatchable(Protocol):

    def dispatch(self, fn: Callable, *args, **kwargs): ...


class Backend(Dispatchable):
    id: Hashable
    requirements: Sequence[Requirement]


# ----------------------------------------------------------------------
# Implementations

class ClassBackend(Backend):
    """Convenience base class for backends also implemented as a class"""

    def dispatch(self, fn: Callable, *args, **kwargs):
        # Dispatch to method on self
        if not hasattr(self, fn.__name__):
            raise NotImplementedError(f'Backend {self.domain}.{self.id} has no "{fn.__name__}" implementation')
        return getattr(self, fn.__name__)(*args, **kwargs)

    @property
    def requirements(self) -> Sequence[Requirement]:
        return []

class MappingBackend(Backend):
    """Convenience base class for backends with implementations provided in a mapping"""

    def __init__(self, id: Hashable, fns: Mapping[str, Callable], reqs: Optional[Sequence[Requirement]] = None):
        self.id = id
        self.fns = fns
        self.requirements = reqs

    def dispatch(self, fn: Callable, *args, **kwargs):
        # Dispatch to method in mapping
        if not fn.__name__ in self.fns:
            raise NotImplementedError(f'Backend "{self.id}" has no "{fn.__name__}" implementation')
        return self.fns[fn.__name__](*args, **kwargs)

class PackageBackend(MappingBackend):
    """Base backend class for package-based implementations"""

    def __init__(self, fns: Mapping[str, Callable]):
        super().__init__(id=self.id, fns=fns, reqs=self.requirements)

class DaskBackend(PackageBackend):
    id = 'dask'
    requirements = [Requirement('dask')]

class NumbaBackend(PackageBackend):
    id = 'numba'
    requirements = [Requirement('numba')]

class CudaBackend(PackageBackend):
    id = 'cuda'
    requirements = [Requirement('numba')]

class NetworkxBackend(PackageBackend):
    id = 'networkx'
    requirements = [Requirement('networkx')]


def is_compatible(backend: Backend):
    """Check if backend meets all requirements

    This is now based solely on installation of packages but may
    expand in the future to OS or system resource constraints as well
    """
    for req in backend.requirements:
        status = check_package(req.package_name)
        if not status.installed or not status.compatible:
            return False
    return True


class Dispatcher(Dispatchable):
    """Default dispatch model"""

    def __init__(self, domain: Union[str, Domain], config: Configuration = None):
        self.domain = Domain(domain)
        self.config = config or global_config
        self.backends = dict()

    def _update_config(self, default='auto'):
        # New backends should result in changes to live, discoverable config attributes
        options = [b.id for b in self.backends.values()]
        key = str(self.domain.append('backend'))
        self.config.register(key, default, f'Options: {options}; default is {default}')

    def register_function(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return self.dispatch(fn, *args, **kwargs)
        return wrapper

    def register_backend(self, backend: Backend) -> Backend:
        self.backends[backend.id] = backend
        self._update_config()
        return backend

    def _process(self, kwargs, default):
        kwargs = dict(kwargs)
        if 'backend' not in kwargs:
            kwargs['backend'] = default
        parts = kwargs.pop('backend').split('/')
        if len(parts) > 1:
            kwargs['backend'] = '/'.join(parts[1:])
        return parts[0], kwargs

    def resolve_backend(self, fn: Callable, *args, **kwargs) -> Backend:
        # Passed parameters get highest priority for backend next
        # to settings in configuration
        backend_id, kwargs = self._process(kwargs, self.config.get(str(self.domain.append('backend'))))

        if backend_id and backend_id != 'auto':
            if backend_id not in self.backends:
                raise ValueError(f'Backend "{backend_id}" not implemented for function {fn.__name__}')
            return self.backends[backend_id], kwargs

        # ** Analyze fn/args/kwargs here **
        # For now, simply return the first compatible backend

        backend = next((b for b in self.backends.values() if is_compatible(b)), None)
        if backend is None:
            raise ValueError(f'No compatible backend found for function "{fn.__name__}" (domain = "{self.domain}"). Check you have installed the required packages.')
        return backend, kwargs

    def dispatch(self, fn: Callable, *args, **kwargs):
        backend, kwargs = self.resolve_backend(fn, *args, **kwargs)
        return backend.dispatch(fn, *args, **kwargs)


# ----------------------------------------------------------------------
# Registration
#
# These functions define the sole interaction points for all 
# frontend/backend coordination across the project

dispatchers: Dict[Domain, Dispatchable] = dict()

def register_function(domain: Union[str, Domain], append: bool=True):
    """Decorator for frontend functionr registration"""
    # For some very odd reason, this throws UnboundLocalError
    # if not aliased to a name other than `domain`
    tmp = Domain(domain)
    def register(fn: Callable):
        domain = tmp
        if append:
            domain = domain.append(fn.__name__)
        if domain not in dispatchers:
            dispatchers[domain] = Dispatcher(domain)
        return dispatchers[domain].register_function(fn)
    return register


def register_backend(domain: Union[str, Domain]):
    """Decorator for backend registration"""
    domain = Domain(domain) 

    def register(backend: Union[Backend, Type[Backend]]):
        if domain not in dispatchers:
            raise NotImplementedError(f'Dispatcher for domain "{domain}" not implemented')
        instance = backend() if isinstance(backend, type) else backend
        dispatchers[domain].register_backend(instance)
        return backend
    return register


def register_backend_function(domain: Union[str, Domain]):
    """Decorator for single-function backend registration"""
    domain = Domain(domain)
    def outer(typ: Type[Backend]):
        def inner(fn: Callable):
            register = register_backend(domain.append(fn.__name__))
            return register(typ(fns={fn.__name__: fn}))
        return inner
    return outer


def dispatches_from(frontend_fn: Callable):
    """Decorator to append doc strings to backend functions"""
    def decorator(backend_fn: Callable):
        # Combine doc strings for backends that may add new parameters or details
        backend_fn.__doc__ = (frontend_fn.__doc__ or '') + '\n' + (backend_fn.__doc__ or '')
        return backend_fn
    return decorator
