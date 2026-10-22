"""Writes demo_examples/_contact_sheet.png (thumbnails of every example, numbered) and strips URLs from the .txt files."""
import re
from pathlib import Path

from PIL import Image, ImageDraw

d = Path("demo_examples")
for t in sorted(d.glob("*.txt")):
    t.write_text(re.sub(r"\s*https?://\S+", "", t.read_text(encoding="utf-8")).strip(), encoding="utf-8")

imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith("_"))
W, H, cols = 300, 230, 5
rows = -(-len(imgs) // cols)
sheet = Image.new("RGB", (cols * W, rows * (H + 24)), "white")
dr = ImageDraw.Draw(sheet)
for i, p in enumerate(imgs):
    im = Image.open(p).convert("RGB"); im.thumbnail((W - 8, H - 8))
    x, y = (i % cols) * W, (i // cols) * (H + 24)
    sheet.paste(im, (x + 4, y + 4)); dr.text((x + 6, y + H), p.stem[:38], fill="black")
sheet.save(d / "_contact_sheet.png"); print(len(imgs), "images")
