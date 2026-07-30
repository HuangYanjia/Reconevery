from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

from PIL import Image, ImageDraw

PREVIEWS = (
    "global_context",
    "measured_anchors",
    "research_assembly",
    "deployment_assembly",
    "object_decision_grid",
    "overlap_heatmap",
    "articulated_snapshot",
)


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("request must be an object")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_preview(path: Path, title: str, width: int, height: int) -> None:
    image = Image.new("RGB", (width, height), (245, 246, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 56, width - 24, height - 24), outline=(60, 75, 90), width=2)
    draw.text((24, 24), title, fill=(20, 30, 40))
    image.save(path, format="PNG", optimize=False, compress_level=9)


def write_diagnostic_glb(path: Path, title: str) -> None:
    document = {
        "asset": {"version": "2.0", "generator": "Reconevery Phase 6B fake worker"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {
                "name": title,
                "extras": {
                    "diagnostic_only": True,
                    "visual_only": True,
                    "sim_ready": False,
                },
            }
        ],
    }
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    padded = encoded + b" " * ((4 - len(encoded) % 4) % 4)
    total = 12 + 8 + len(padded)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(padded), 0x4E4F534A)
        + padded
    )


def assemble(request_path: Path, input_root: Path, output_dir: Path) -> None:
    request = read_json(request_path)
    mode = str(request.get("fake_mode", "success"))
    if mode == "timeout":
        time.sleep(60)
    if mode == "worker_modifying_upstream_assets":
        (input_root / str(request["assembly_plan_path"])).write_text(
            "modified\n",
            encoding="utf-8",
        )
        return
    configuration = request["preview_configuration"]
    if not isinstance(configuration, dict):
        raise ValueError("preview configuration must be an object")
    width = int(configuration["image_width"])
    height = int(configuration["image_height"])
    previews = output_dir / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    for name in PREVIEWS:
        write_preview(previews / f"{name}.png", name.replace("_", " ").title(), width, height)
    assets = output_dir / "preview_assets"
    write_diagnostic_glb(assets / "research_scene.glb", "research visual preview")
    write_diagnostic_glb(assets / "deployment_scene.glb", "deployment visual preview")
    material_before = 2
    material_after = 1 if mode == "preview_material_loss" else material_before
    write_json(
        output_dir / "preview_manifest.json",
        {
            "schema_version": "0.1.0",
            "preview_paths": {name: f"assembly/previews/{name}.png" for name in PREVIEWS},
            "preview_asset_paths": {
                "research": "assembly/preview_assets/research_scene.glb",
                "deployment": "assembly/preview_assets/deployment_scene.glb",
            },
            "material_count_before": material_before,
            "material_count_after": material_after,
            "texture_count_before": 0,
            "texture_count_after": 0,
            "representation_warnings": (
                ["preview worker could not preserve every material"]
                if material_after < material_before
                else []
            ),
            "diagnostic_only": True,
            "source_geometry_modified": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["assemble", "healthcheck"])
    parser.add_argument("--request", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "healthcheck":
        print(json.dumps({"ok": True, "worker": "fake_scene_assembly"}))
        return
    if args.request is None or args.input_root is None or args.output_dir is None:
        parser.error("assemble requires --request, --input-root, and --output-dir")
    assemble(args.request, args.input_root, args.output_dir)


if __name__ == "__main__":
    main()
