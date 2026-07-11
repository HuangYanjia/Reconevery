from __future__ import annotations

from pathlib import Path
from typing import Any


class BadParameter(Exception):
    pass


class Typer:
    def __init__(self, help: str = ""):
        self.commands = {}
        self.groups = {}

    def command(self, name: str | None = None, **kw: Any):
        def deco(fn: Any) -> Any:
            self.commands[name or fn.__name__.replace("_", "-")] = fn
            return fn

        return deco

    def add_typer(self, app: Typer, name: str) -> None:
        self.groups[name] = app


class Option:
    def __init__(self, default: Any = None, *a: Any, **k: Any):
        self.default = default


class Argument(Option):
    pass


def echo(msg: Any = "") -> None:
    print(msg)
