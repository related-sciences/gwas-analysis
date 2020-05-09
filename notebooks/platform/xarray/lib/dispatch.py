from __future__ import annotations
import functools
from typing import Hashable, Callable, Union, Sequence, Dict, Type
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

    def requirements(self) -> Sequence[Requirement]: ...


# ----------------------------------------------------------------------
# Implementations

class ClassBackend(Backend):
    """Convenience base class for backends also implemented as a class

    Alternatives could include backends that are implemented as Descriptors or Mappings,
    but classes are likely more familiar/flexible for most devs wishing to contribute or users
    looking to add custom backends (so this may be all that's necessary)
    """

    def dispatch(self, fn: Callable, *args, **kwargs):
        # Dispatch to method on self
        if not hasattr(self, fn.__name__):
            raise NotImplementedError(f'Backend {self.domain}.{self.id} has no "{fn.__name__}" implementation')
        return getattr(self, fn.__name__)(*args, **kwargs)

    def requirements(self) -> Sequence[Requirement]:
        # Return no requirements by default
        return []


def is_compatible(backend: Backend):
    """Check if backend meets all requirements

    This is now based solely on installation of packages but may
    expand in the future to OS or system resource constraints as well
    """
    for req in backend.requirements():
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

    def resolve(self, fn: Callable, *args, **kwargs) -> Backend:
        # Passed parameters get highest priority
        backend_id = kwargs.get('backend')

        # Followed by configuration
        backend_id = backend_id or self.config.get(str(self.domain.append('backend')))
        if backend_id and backend_id != 'auto':
            return self.backends[backend_id]

        # And then automatic selection/validation:

        # ** Analyze fn/args/kwargs here **
        # For now, simply return the first compatible backend
        backend = next((b for b in self.backends.values() if is_compatible(b)), None)
        if backend is None:
            raise ValueError(f'No suitable backend found for function {fn.__name__} (domain = {self.domain})')
        return backend

    def dispatch(self, fn: Callable, *args, **kwargs):
        backend = self.resolve(fn, *args, **kwargs)
        kwargs.pop('backend', None)  # Pop this off since implementations cannot expect it
        return backend.dispatch(fn, *args, **kwargs)


# ----------------------------------------------------------------------
# Registry
#
# These functions define the sole interaction points for all 
# frontend/backend coordination across the project

dispatchers: Dict[Domain, Dispatchable] = dict()

def register_function(domain):
    domain = Domain(domain)
    if domain not in dispatchers:
        dispatchers[domain] = Dispatcher(domain)
        
    def register(fn: Callable):
        return dispatchers[domain].register_function(fn)
    return register


def register_backend(domain):
    domain = Domain(domain) 
    if domain not in dispatchers:
        raise NotImplementedError('Dispatcher for domain {domain} not implemented')

    def register(backend: Union[Backend, Type[Backend]]):
        instance = backend() if isinstance(backend, type) else backend
        dispatchers[domain].register_backend(instance)
        return backend
    return register
