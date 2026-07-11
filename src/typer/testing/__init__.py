from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace


class CliRunner:
    def invoke(self, app, args):
        out = StringIO()
        code = 0
        try:
            with redirect_stdout(out), redirect_stderr(out):
                cmd = args[0]
                rest = args[1:]
                fn = app.commands[cmd]
                kwargs = {}
                pos = []
                i = 0
                while i < len(rest):
                    if rest[i].startswith("--"):
                        key = rest[i][2:].replace("-", "_")
                        if i + 1 < len(rest) and not rest[i + 1].startswith("--"):
                            kwargs[key] = (
                                Path(rest[i + 1])
                                if key in {"input", "config", "run_dir"}
                                or rest[i + 1].endswith(".json")
                                else rest[i + 1]
                            )
                            i += 2
                        else:
                            kwargs[key] = True
                            i += 1
                    else:
                        pos.append(Path(rest[i]))
                        i += 1
                fn(*pos, **kwargs)
        except Exception as e:
            code = 1
            out.write(str(e))
        return SimpleNamespace(exit_code=code, output=out.getvalue())
