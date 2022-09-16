import functools
import inspect
from collections import namedtuple


def arguments_to_config(init):
    """
    A convenience wrapper that automatically registers received arguments.
    """

    @functools.wraps(init)
    def init_wrapper(self, *args, **kwargs):
        init(self, *args, **kwargs)

        signature = inspect.signature(init)
        parameters = {
            name: p.default for i, (name, p) in enumerate(signature.parameters.items()) if i > 0
        }

        _kwargs = {}
        for arg, name in zip(args, parameters.keys()):
            _kwargs[name] = arg

        _kwargs.update({
            kw: kwargs.get(kw, default)
            for kw, default in parameters.items()
            if kw not in _kwargs
        })

        Config = namedtuple(f"{self.__class__.__name__}Config", [k for k in _kwargs.keys()])
        config = Config(**_kwargs)

        setattr(self, "config", config)

    return init_wrapper
