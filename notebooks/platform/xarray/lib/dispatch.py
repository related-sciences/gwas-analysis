from __future__ import annotations
import functools
from typing import Hashable, Callable, Union, Sequence
from typing_extensions import Protocol
from .config import Configuration
from .config import config as global_config
from .compat import Requirement, check_package


class Domain(tuple):
    sep = '.'

    def __new__(cls, value: Union[str, Sequence[str]]):
        if isinstance(value, str):
            value = value.split(Domain.sep)
        return super().__new__(cls, tuple(value))

    def append(self, other: Union[str, Sequence[str]]) -> Domain:
        return Domain(self + Domain(other))

    def startswith(self, other: Sequence[str]):
        if len(self) < len(other):
            return False
        return all([p == self[i] for i, p in enumerate(other)])

    def __str__(self):
        return Domain.sep.join(self)


class Dispatchable(Protocol):

    def dispatch(self, fn: Callable, *args, **kwargs): ...


class Frontend(Dispatchable):
    domain: Domain


class Backend(Dispatchable):
    domain: Domain
    id: Hashable

    def requirements(self) -> Sequence[Requirement]: ...


class ClassBackend(Backend):

    def dispatch(self, fn: Callable, *args, **kwargs):
        # Dispatch to method on self
        if not hasattr(self, fn.__name__):
            raise NotImplementedError(f'Backend {self.domain}.{self.id} has no "{fn.__name__}" implementation')
        return getattr(self, fn.__name__)(*args, **kwargs)

    def requirements(self) -> Sequence[Requirement]:
        return []


def is_compatible(backend: str):
    for req in backend.requirements:
        status = check_package(req.package_name)
        if not status.installed or not status.compatible:
            return False
    return True


class FrontendDispatcher(Frontend):

    def __init__(self, domain: str, config: Configuration = None):
        self.domain = Domain(domain)
        self.config = config or global_config
        self.backends = dict()

    def _update_config(self, default='auto'):
        options = [b.id for b in self.backends.values()]
        key = str(self.domain.append('backend'))
        self.config.register(key, default, f'Options: {options}; default is {default}')

    def register(self, backend: Backend) -> None:
        if not backend.domain == self.domain:
            raise ValueError('Backend with domain {backend.domain} not compatible with frontend domain {self.domain}')
        self.backends[backend.id] = backend
        self._update_config()

    def resolve(self, fn: Callable, *args, **kwargs) -> Backend:
        backend_id = kwargs.get('backend')
        backend_id = backend_id or self.config.get(str(self.domain.append('backend')))
        if backend_id and backend_id != 'auto':
            return self.backends[backend_id]
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

    def add(self, fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return self.dispatch(fn, *args, **kwargs)
        return wrapper
