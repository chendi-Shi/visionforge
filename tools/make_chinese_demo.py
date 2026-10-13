"""Create a clearly synthetic Chinese receipt for manual extraction checking."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path is None:
        raise SystemExit("Install a Chinese font (Microsoft YaHei or Noto Sans CJK) first.")
    normal = ImageFont.truetype(str(font_path), 30)
    heading = ImageFont.truetype(str(font_path), 42)
    picture = Image.new("RGB", (720, 480), "white")
    draw = ImageDraw.Draw(picture)
    draw.rounded_rectangle((18, 18, 702, 462), radius=15, outline="#26635c", width=3)
    draw.text((245, 40), "晨光果铺", font=heading, fill="#1d4e47")
    draw.text((55, 115), "商品        单价       数量", font=normal, fill="#252525")
    draw.line((55, 168, 665, 168), fill="#9bb6b1", width=2)
    draw.text((55, 193), "苹果        3.00元       2", font=normal, fill="#252525")
    draw.text((55, 251), "香蕉        2.00元       1", font=normal, fill="#252525")
    draw.line((55, 316, 665, 316), fill="#9bb6b1", width=2)
    draw.text((55, 347), "合计：8.00元", font=heading, fill="#1d4e47")
    (ROOT / "examples").mkdir(exist_ok=True)
    picture.save(ROOT / "examples" / "receipt_zh.png")
    print("Created synthetic Chinese receipt (manual smoke test only).")


if __name__ == "__main__":
    main()
