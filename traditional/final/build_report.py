"""
Build a visual HTML report: each cropped text image with its predicted text
shown underneath it.

Reads a predictions CSV (columns: image, model, prediction) — the same format
``predict.py --save`` produces — groups the rows by image, and writes a single
self-contained HTML page. Images are embedded as base64 data URIs so the page
can be opened or shared anywhere without the original files.

Usage
-----
    # Default: read final/predictions.csv, write final/report.html
    python final/build_report.py

    # Custom input / output
    python final/build_report.py --csv final/predictions.csv --out final/report.html
"""

import argparse
import base64
import csv
import html
import mimetypes
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "predictions.csv"
DEFAULT_OUT = HERE / "report.html"


def embed_image(image_path: Path) -> str | None:
    """Return a base64 data URI for the image, or None if it can't be read."""
    try:
        data = image_path.read_bytes()
    except OSError:
        return None
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def read_rows(csv_path: Path):
    """Group CSV rows by image, preserving file order: {image: [(model, pred), ...]}."""
    grouped: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = row.get("image", "").strip()
            if not image:
                continue
            model = (row.get("model") or "").strip()
            prediction = (row.get("prediction") or "").strip()
            grouped.setdefault(image, []).append((model, prediction))
    return grouped


def render_card(image: str, predictions: list[tuple[str, str]]) -> str:
    """Render one crop: the image on top, its prediction(s) underneath."""
    path = Path(image)
    uri = embed_image(path)
    if uri is not None:
        img_html = f'<img src="{uri}" alt="{html.escape(path.name)}">'
    else:
        img_html = f'<div class="missing">image not found:<br>{html.escape(image)}</div>'

    rows = []
    show_model = len(predictions) > 1 or (predictions and predictions[0][0])
    for model, prediction in predictions:
        text = html.escape(prediction) if prediction else '<span class="empty">(empty)</span>'
        model_tag = f'<span class="model">{html.escape(model)}</span>' if show_model else ""
        rows.append(f'<div class="pred">{model_tag}<span class="text">{text}</span></div>')

    return (
        '<figure class="card">'
        f'<div class="crop">{img_html}</div>'
        f'<figcaption>{"".join(rows)}</figcaption>'
        f'<div class="filename">{html.escape(path.name)}</div>'
        "</figure>"
    )


def build_html(grouped) -> str:
    cards = "\n".join(render_card(img, preds) for img, preds in grouped.items())
    total = len(grouped)
    return f"""<!DOCTYPE html>
<html lang="km">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR Report</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    font-family: "Noto Sans Khmer", "Khmer OS", system-ui, sans-serif;
    margin: 0; padding: 24px; background: #f4f5f7; color: #1c1e21;
  }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .meta {{ color: #65676b; font-size: 13px; margin-bottom: 20px; }}
  .grid {{
    display: grid; gap: 16px;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  }}
  .card {{
    margin: 0; background: #fff; border: 1px solid #dadde1; border-radius: 10px;
    padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,.06); display: flex; flex-direction: column;
  }}
  .crop {{
    background: #fafafa; border: 1px solid #eee; border-radius: 6px;
    padding: 8px; text-align: center; min-height: 48px;
    display: flex; align-items: center; justify-content: center;
  }}
  .crop img {{ max-width: 100%; height: auto; image-rendering: auto; }}
  .missing {{ color: #b00020; font-size: 12px; }}
  figcaption {{ margin-top: 10px; }}
  .pred {{ display: flex; gap: 8px; align-items: baseline; padding: 3px 0; }}
  .pred + .pred {{ border-top: 1px dashed #eee; }}
  .model {{
    flex: 0 0 auto; font-size: 11px; color: #65676b; background: #eef0f3;
    border-radius: 4px; padding: 1px 6px; white-space: nowrap;
  }}
  .text {{ font-size: 20px; line-height: 1.5; word-break: break-word; }}
  .empty {{ color: #b0b3b8; font-style: italic; font-size: 14px; }}
  .filename {{ margin-top: 8px; font-size: 11px; color: #90949c; word-break: break-all; }}
</style>
</head>
<body>
  <h1>OCR Report</h1>
  <div class="meta">{total} cropped image(s) — each crop shown with its predicted text below.</div>
  <div class="grid">
{cards}
  </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Build an HTML page of crops + predicted text")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Predictions CSV (image,model,prediction)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output HTML path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    grouped = read_rows(csv_path)
    if not grouped:
        raise SystemExit(f"No rows found in {csv_path}")

    out_path = Path(args.out)
    out_path.write_text(build_html(grouped), encoding="utf-8")
    print(f"Wrote {out_path}  ({len(grouped)} crops)")


if __name__ == "__main__":
    main()
