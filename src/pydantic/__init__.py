from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, get_args, get_origin, get_type_hints


def Field(default: Any = None, default_factory: Any = None, **_: Any) -> Any:
    return default_factory() if default_factory else default


def ConfigDict(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def field_validator(*_: str, **__: Any):
    def deco(fn: Any) -> Any:
        return fn

    return deco


class ValidationError(Exception):
    pass


class BaseModel:
    model_config: dict[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        anns = self._annotations()
        for k in data:
            if k not in anns and self.model_config.get("extra") == "forbid":
                raise ValidationError(f"extra field {k}")
        for k, t in anns.items():
            if k in data:
                v = data[k]
            elif hasattr(self.__class__, k):
                v = getattr(self.__class__, k)
            else:
                raise ValidationError(f"missing field {k}")
            setattr(self, k, self._coerce(v, t))
        if self.__class__.__name__ == "SceneIR":
            objs = getattr(self, "objects", [])
            ids = [getattr(o, "object_id", None) for o in objs]
            if len(ids) != len(set(ids)):
                raise ValidationError("object_id values must be unique")

    @classmethod
    def _annotations(cls) -> dict[str, Any]:
        out = {}
        for c in reversed(cls.mro()):
            try:
                out.update(get_type_hints(c))
            except Exception:
                out.update(getattr(c, "__annotations__", {}))
        out.pop("model_config", None)
        return out

    @classmethod
    def _coerce(cls, v: Any, t: Any) -> Any:
        origin = get_origin(t)
        args = get_args(t)
        if v is None:
            return None
        if origin is list:
            return [cls._coerce(x, args[0]) for x in v]
        if origin is tuple:
            return tuple(v)
        if origin is dict:
            kt, vt = args if len(args) == 2 else (Any, Any)
            return {k: cls._coerce(val, vt) for k, val in dict(v).items()}
        if (
            origin is not None
            and str(origin).endswith("UnionType")
            or origin is getattr(__import__("typing"), "Union", None)
        ):
            for a in args:
                if a is type(None):
                    continue
                try:
                    return cls._coerce(v, a)
                except Exception:
                    pass
            return v
        try:
            if isinstance(t, type) and issubclass(t, BaseModel):
                return v if isinstance(v, t) else t(**v)
            if isinstance(t, type) and issubclass(t, Enum):
                return t(v)
            if t is datetime and isinstance(v, str):
                return datetime.fromisoformat(v)
        except TypeError:
            pass
        return v

    @classmethod
    def model_validate(cls, data: Any) -> Any:
        return data if isinstance(data, cls) else cls(**data)

    @classmethod
    def model_validate_json(cls, text: str) -> Any:
        return cls.model_validate(json.loads(text))

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        def conv(x: Any) -> Any:
            if isinstance(x, BaseModel):
                return x.model_dump(mode=mode)
            if isinstance(x, Enum):
                return x.value
            if isinstance(x, datetime):
                return x.isoformat()
            if isinstance(x, list):
                return [conv(i) for i in x]
            if isinstance(x, tuple):
                return [conv(i) for i in x]
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            return x

        return {k: conv(getattr(self, k)) for k in self._annotations() if hasattr(self, k)}

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        return {"title": cls.__name__, "type": "object"}
