import functools
import inspect
from collections import namedtuple
from typing import Any, Dict, List, Set, Tuple

import torch


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


def torch_float_or_long(method):
    def cast_if_torch(cls, *args, **kwargs):
        def _cast_if_torch(arg: Any):
            if isinstance(arg, torch.Tensor):
                if arg.dtype in {torch.float64, torch.double}:
                    arg = arg.float()
                elif arg.dtype in {torch.int32, torch.int}:
                    arg = arg.long()
                return arg

            if isinstance(arg, (List, Tuple, Set)):
                return type(arg)(_cast_if_torch(_arg) for _arg in arg)
            if isinstance(arg, Dict):
                return type(arg)(**{kw: _cast_if_torch(_arg) for kw, _arg in arg.items()})

            return arg

        return method(cls, *_cast_if_torch(args), **_cast_if_torch(kwargs))

    return cast_if_torch
