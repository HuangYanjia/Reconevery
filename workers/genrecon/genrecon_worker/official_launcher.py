from __future__ import annotations

import argparse
import inspect
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def install_iphone_skip_cleaning_compat(chunker_class: type[Any]) -> bool:
    """Add the CLI-declared skip flag to the pinned IphoneChunker constructor."""
    original_init = chunker_class.__init__
    if "skip_point_cleaning" in inspect.signature(original_init).parameters:
        return False

    def compatible_init(
        self: Any,
        *args: object,
        skip_point_cleaning: bool = False,
        **kwargs: object,
    ) -> None:
        original_init(self, *args, **kwargs)
        self.skip_point_cleaning = skip_point_cleaning

    chunker_class.__init__ = compatible_init
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the pinned official GenRecon CLI with narrow compatibility fixes."
    )
    parser.add_argument("--script", type=Path, required=True)
    arguments, official_arguments = parser.parse_known_args(argv)
    if official_arguments[:1] == ["--"]:
        official_arguments = official_arguments[1:]

    script = arguments.script.resolve()
    if script.name != "reconstruct_scene.py" or not script.is_file():
        parser.error("--script must reference the official reconstruct_scene.py")
    sys.path.insert(0, str(script.parent))

    from inference.get_chunks import IphoneChunker

    install_iphone_skip_cleaning_compat(IphoneChunker)
    sys.argv = [str(script), *official_arguments]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
