# Movie Finder

A Flask web app that recommends movies from natural-language text queries using TF-IDF and cosine-similarity based content matching.

## Local run

```bash
pip install -r requirements.txt
gunicorn flask_app:app --timeout 180 --workers 1
```
