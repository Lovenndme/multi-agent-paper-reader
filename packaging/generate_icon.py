"""Generate deterministic multi-size Windows branding assets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SIZE = 1024


def generate() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Soft blue application tile.
    for inset in range(0, 48):
        ratio = inset / 48
        color = (
            int(23 + 12 * ratio),
            int(105 + 40 * ratio),
            int(224 + 18 * ratio),
            255,
        )
        draw.rounded_rectangle(
            (inset, inset, SIZE - inset - 1, SIZE - inset - 1),
            radius=220 - inset * 2,
            fill=color,
        )

    # Paper sheet with a folded corner.
    paper = [(270, 188), (632, 188), (770, 326), (770, 790), (270, 790)]
    draw.polygon(paper, fill=(248, 252, 255, 255))
    draw.line(paper + [paper[0]], fill=(221, 237, 255, 255), width=18, joint="curve")
    draw.polygon([(632, 188), (632, 326), (770, 326)], fill=(201, 226, 255, 255))
    draw.line([(632, 188), (632, 326), (770, 326)], fill=(152, 199, 255, 255), width=15)

    # Text lines.
    for y, width in ((395, 250), (470, 330), (545, 220)):
        draw.rounded_rectangle((350, y, 350 + width, y + 26), radius=13, fill=(80, 138, 213, 255))

    # Agent network on the lower paper area.
    nodes = [(408, 665), (520, 625), (632, 665), (520, 735)]
    links = [(0, 1), (1, 2), (0, 3), (2, 3), (1, 3)]
    for start, end in links:
        draw.line([nodes[start], nodes[end]], fill=(23, 105, 224, 255), width=16)
    for x, y in nodes:
        draw.ellipse((x - 31, y - 31, x + 31, y + 31), fill=(23, 105, 224, 255))
        draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=(244, 250, 255, 255))

    png_path = ASSETS / "paper-reader.png"
    ico_path = ASSETS / "paper-reader.ico"
    image.save(png_path, optimize=True)
    image.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(png_path)
    print(ico_path)


if __name__ == "__main__":
    generate()
