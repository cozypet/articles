#!/usr/bin/env python3
"""Turn article markdown into plain pages that Medium's import tool can read.

Usage:  python3 tools/build_pages.py            (builds every folder that has a pages.json)
        python3 tools/build_pages.py kvcache    (builds one folder)

Each article folder holds:
  pages.json   site address, folder name, link placeholders, list of pages
  source/      the markdown files
  images/      the card images, referenced in markdown as images/NN.png
Output: [folder]/[out]/index.html, with every image as an absolute address on GitHub Pages.
"""
import html
import json
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent

CSS = """
body { max-width: 740px; margin: 0 auto; padding: 40px 20px; font-family: Georgia, serif; font-size: 18px; line-height: 1.65; color: #141413; background: #faf9f5; }
h1 { font-size: 34px; line-height: 1.2; }
h2 { font-size: 26px; margin-top: 44px; }
img { max-width: 100%; height: auto; display: block; margin: 28px auto; }
em.sub { font-size: 20px; color: #5c5b55; }
a { color: #d97757; }
""".strip()

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="author" content="Han HELOIR YAN, Ph.D.">
<meta name="description" content="{subtitle}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{subtitle}">
<meta property="og:type" content="article">
<style>
{css}
</style>
</head>
<body>

{body}

</body>
</html>
"""


def build_page(folder: Path, cfg: dict, page: dict) -> dict:
    src = (folder / page["source"]).read_text(encoding="utf-8")
    base = f'{cfg["site"]}/{cfg["folder"]}'

    # Link placeholders: a real link when the address is known, plain words when it is not.
    for key, url in cfg.get("links", {}).items():
        if url:
            src = src.replace(f"]({key})", f"]({url})")
        else:
            src = re.sub(r"\[([^\]]+)\]\(" + re.escape(key) + r"\)", r"\1", src)
    left = re.findall(r"\]\((PART_\d+_URL)\)", src)
    if left:
        raise SystemExit(f'{page["source"]}: placeholder {left[0]} is missing from pages.json')

    body = markdown.markdown(src, extensions=[], output_format="html")

    # Images stand alone (not inside a paragraph) and use absolute addresses.
    body = re.sub(r"<p>\s*(<img [^>]+>)\s*</p>", r"\1", body)
    body = re.sub(r'src="images/', f'src="{base}/images/', body)
    body = body.replace(" />", ">")

    title = html.unescape(re.search(r"<h1>(.*?)</h1>", body, re.S).group(1))
    m = re.search(r"</h1>\s*<p><em>(.*?)</em></p>", body, re.S)
    subtitle = html.unescape(m.group(1)) if m else ""
    if m:
        body = body.replace(m.group(0), f'</h1>\n<p><em class="sub">{m.group(1)}</em></p>', 1)

    out_dir = folder / page["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(
        PAGE.format(title=html.escape(title, quote=True), subtitle=html.escape(subtitle, quote=True), css=CSS, body=body),
        encoding="utf-8",
    )
    imgs = re.findall(r'<img [^>]*src="([^"]+)"', body)
    return {"title": title, "subtitle": subtitle, "url": f'{base}/{page["out"]}/', "images": imgs}


def build_folder(folder: Path) -> list:
    cfg = json.loads((folder / "pages.json").read_text(encoding="utf-8"))
    return [build_page(folder, cfg, p) for p in cfg["pages"]]


def build_home(all_pages: list) -> None:
    items = "\n".join(
        f'<li><a href="{p["url"]}">{html.escape(p["title"])}</a><br><small>{html.escape(p["subtitle"])}</small><br><code>{p["url"]}</code></li>'
        for p in all_pages
    )
    (ROOT / "index.html").write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Article pages</title>
<style>
{CSS}
li {{ margin-bottom: 22px; }}
code {{ font-size: 14px; color: #5c5b55; word-break: break-all; }}
</style>
</head>
<body>
<h1>Article pages</h1>
<p>Source pages for Han HELOIR YAN's Medium articles. Paste an address below into medium.com/p/import.</p>
<ul>
{items}
</ul>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    wanted = sys.argv[1:]
    folders = sorted(p.parent for p in ROOT.glob("*/pages.json"))
    all_pages = []
    for folder in folders:
        pages = build_folder(folder)
        all_pages += pages
        if not wanted or folder.name in wanted:
            for p in pages:
                print(f'{p["url"]}  ({len(p["images"])} images)  {p["title"]}')
    build_home(all_pages)


if __name__ == "__main__":
    main()
