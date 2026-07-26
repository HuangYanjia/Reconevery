from pathlib import Path

from PIL import Image, ImageDraw


def write_preview(path: Path, title: str, lines: list[str]) -> None:
    image = Image.new("RGB", (960, 540), (246, 247, 249))
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), title, fill=(25, 35, 45))
    for index, line in enumerate(lines):
        draw.text((20, 60 + index * 24), line, fill=(50, 65, 80))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=9)
