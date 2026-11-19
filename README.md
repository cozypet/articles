# Article pages

Source pages for the Medium articles of Han HELOIR YAN, Ph.D. (Signal Over Noise).

Each article lives in its own folder. A page built here is read once by Medium's import tool, which brings the text and the images into a draft.

## Layout

```
[article]/pages.json     site address, link placeholders, list of pages
[article]/source/        the article text in markdown, plus the diagram source
[article]/images/        the diagram cards as PNG
[article]/[page]/        the built page that Medium imports
tools/build_pages.py     turns the markdown into pages
```

## Build

```
pip install markdown
python3 tools/build_pages.py
```

The script prints the address of every page. Paste one into medium.com/p/import.

## Add an article

1. Create a folder with `source/`, `images/` and a `pages.json` (copy the one in `kvcache/`).
2. In the markdown, reference images as `images/NN.png`.
3. Run the build, commit, push.
