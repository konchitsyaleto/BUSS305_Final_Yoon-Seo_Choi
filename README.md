# Movie Finder

Movie Finder is available as a GitHub Pages static website through `index.html`.

The static page loads `assets/data/movies.json` in the browser and ranks movies from natural-language text queries.

## GitHub Pages

1. Push this repository to GitHub.
2. Open repository settings.
3. Go to Pages.
4. Set the source to `Deploy from a branch`.
5. Select the `main` branch and `/ (root)`.
6. Save.

## Local run

For the GitHub Pages version, open `index.html` through a local static server.

```bash
python -m http.server 8000
```

The original Flask version is still included:

```bash
pip install -r requirements.txt
gunicorn flask_app:app --timeout 180 --workers 1
```
