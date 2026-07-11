from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

NAME = "recon2sim"
VERSION = "0.1.0"
DIST = f"{NAME}-{VERSION}.dist-info"


def _wheel(wheel_directory: str) -> str:
    out = Path(wheel_directory) / f"{NAME}-{VERSION}-py3-none-any.whl"
    records: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in Path("src").rglob("*.py"):
            arc = path.relative_to("src").as_posix()
            data = path.read_bytes()
            zf.writestr(arc, data)
            records.append((arc, data))
        files = {
            f"{DIST}/METADATA": f"Metadata-Version: 2.3\nName: {NAME}\nVersion: {VERSION}\nRequires-Python: >=3.12\n".encode(),
            f"{DIST}/WHEEL": b"Wheel-Version: 1.0\nGenerator: recon2sim-build\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            f"{DIST}/entry_points.txt": b"[console_scripts]\nrecon2sim = recon2sim.cli:main\n",
        }
        for arc, data in files.items():
            zf.writestr(arc, data)
            records.append((arc, data))
        rows = []
        for arc, data in records:
            digest = hashlib.sha256(data).digest()
            import base64

            rows.append(
                f"{arc},sha256={base64.urlsafe_b64encode(digest).rstrip(b'=').decode()},{len(data)}"
            )
        rows.append(f"{DIST}/RECORD,,")
        zf.writestr(f"{DIST}/RECORD", "\n".join(rows) + "\n")
    return out.name


def build_wheel(wheel_directory: str, config_settings=None, metadata_directory=None) -> str:
    return _wheel(wheel_directory)


def build_editable(wheel_directory: str, config_settings=None, metadata_directory=None) -> str:
    return _wheel(wheel_directory)


def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    return []


def get_requires_for_build_editable(config_settings=None) -> list[str]:
    return []
