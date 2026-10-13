"""Draw our own three synthetic QA smoke examples; no external dataset involved."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "examples"


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def create_examples(destination: Path = OUTPUT_DIR) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (720, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 22), "Count the red circles", font=font(34), fill="#162238")
    for x, y in ((100, 160), (295, 160), (100, 345), (295, 345)):
        draw.ellipse((x - 52, y - 52, x + 52, y + 52), fill="#de3346")
    draw.rectangle((485, 112, 593, 220), fill="#286ed2")
    draw.polygon(((540, 294), (476, 397), (604, 397)), fill="#28a667")
    image.save(destination / "shapes.png")

    image = Image.new("RGB", (720, 480), "#eff6f2")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((32, 32, 688, 448), radius=16, fill="white", outline="#254d40", width=4)
    draw.rectangle((34, 34, 686, 145), fill="#254d40")
    draw.text((130, 63), "MAPLE CAFE", font=font(52), fill="white")
    draw.text((70, 190), "COFFEE", font=font(45), fill="#202020")
    draw.text((490, 190), "$3.50", font=font(45), fill="#202020")
    draw.line((70, 265, 648, 265), fill="#c4d2cb", width=2)
    draw.text((70, 310), "TEA", font=font(45), fill="#202020")
    draw.text((490, 310), "$2.00", font=font(45), fill="#202020")
    image.save(destination / "cafe.png")

    image = Image.new("RGB", (720, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 24), "Weekly units sold", font=font(34), fill="#162238")
    baseline, top, left, right = 382, 105, 90, 655
    for value in range(0, 31, 10):
        y = baseline - (baseline - top) * value / 30
        draw.line((left, y, right, y), fill="#d6dde6", width=2)
        draw.text((40, y - 15), str(value), font=font(23), fill="#485166")
    for label, value, x, color in (("A", 10, 160, "#3285d1"), ("B", 20, 335, "#e29034"), ("C", 30, 510, "#38a483")):
        y = baseline - (baseline - top) * value / 30
        draw.rectangle((x, y, x + 75, baseline), fill=color)
        draw.text((x + 20, y + 12), str(value), font=font(26), fill="white")
        draw.text((x + 24, baseline + 15), label, font=font(28), fill="#162238")
    image.save(destination / "bars.png")

    examples = [
        {"id": "synthetic-shape-count", "image": "shapes.png", "question": "How many red circles are shown? Answer with one digit only.", "accepted_answers": ["4"]},
        {"id": "synthetic-menu-price", "image": "cafe.png", "question": "What is the price of COFFEE at MAPLE CAFE? Answer with the price only, including the dollar sign.", "accepted_answers": ["$3.50"]},
        {"id": "synthetic-bar-reading", "image": "bars.png", "question": "How many units were sold for B? Answer with the number only.", "accepted_answers": ["20"]},
    ]
    (destination / "manifest.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    return examples


if __name__ == "__main__":
    examples = create_examples()
    print(f"Created {len(examples)} synthetic smoke examples in {OUTPUT_DIR}")
    print("These are self-drawn smoke tests, not a public benchmark or a measure of real-world generalization.")
