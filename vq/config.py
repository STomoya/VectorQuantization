import inspect
import json
from functools import wraps


class ConfigMixin:
    _repr_config: dict

    def __init_subclass__(cls) -> None:
        cls.__init__ = _register_to_config(cls.__init__)

    def save_config(self, filename: str):
        config = self._repr_config
        with open(filename, 'w') as fp:
            json.dump(config, fp, indent=2)

    @classmethod
    def from_config(cls, filename: str):
        with open(filename, 'r') as fp:
            kwargs = json.load(fp)
        obj = cls(**kwargs)
        return obj


def _register_to_config(init):
    @wraps(init)
    def inner_init(self: ConfigMixin, *args, **kwargs):
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {name: p.default for i, (name, p) in enumerate(signature.parameters.items()) if i > 0}
        for arg, name in zip(args, parameters.keys()):  # noqa: B905
            new_kwargs[name] = arg

        new_kwargs.update({k: kwargs.get(k, default) for k, default in parameters.items() if k not in new_kwargs})

        self._repr_config = new_kwargs
        init(self, *args, **kwargs)

    return inner_init
