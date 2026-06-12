from html import escape

from flask import Flask, request, render_template_string, url_for
import pandas as pd

from movie_recommender import recommend_movies


app = Flask(__name__)


def safe_text(value, fallback=""):
    if isinstance(value, str) and value.strip():
        return value.strip()
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def format_rating(value):
    if pd.isna(value) or value == "":
        return "N/A"
    return f"{float(value):.1f}"


def format_year(value):
    if pd.isna(value) or value == "":
        return "N/A"
    return str(int(float(value)))


def score_color(score):
    if score >= 85:
        return "#9ec5ff"
    if score >= 70:
        return "#a7e8c2"
    if score >= 45:
        return "#f8e7a1"
    return "#f2a6a6"


def link_or_text(url, label):
    if isinstance(url, str) and url.strip():
        return f'<a class="rating-link" href="{escape(url)}" target="_blank">{label}</a>'
    return f'<span class="rating-link">{label}</span>'


def movie_card(row):
    title = escape(safe_text(row["movie_title"], "Untitled"))
    year = escape(format_year(row["release_year"]))
    countries = escape(safe_text(row["display_countries"], "Unknown country"))
    director = escape(safe_text(row["directors"], "Director N/A"))
    poster_url = safe_text(row["poster_url"])
    overview = escape(safe_text(row["display_overview"], "No overview available."))
    score = int(row["match_score"])
    color = score_color(score)

    imdb = link_or_text(
        safe_text(row["imdb_url"]),
        f'<span class="star-icon">★</span> {format_rating(row["tmdb_vote_average"])}',
    )
    tomato = link_or_text(
        safe_text(row["rotten_tomatoes_url"]),
        f"🍅 {format_rating(row['tomatometer_rating'])}",
    )
    popcorn = link_or_text(
        safe_text(row["rotten_tomatoes_url"]),
        f"🍿 {format_rating(row['audience_rating'])}",
    )

    poster = (
        f'<img class="poster" src="{escape(poster_url)}" alt="{title} poster">'
        if poster_url
        else '<div class="poster poster-fallback">No Poster</div>'
    )

    keywords = row["match_keywords"] if isinstance(row["match_keywords"], list) else []
    tags = "".join(f'<span class="tag">#{escape(str(word))}</span>' for word in keywords[:8])

    return f"""
    <details class="movie-card">
        <summary>
            <div class="poster-wrap">{poster}</div>
            <div class="movie-main">
                <div class="movie-title">{title}</div>
                <div class="movie-facts">
                    <span>{year}</span>
                    <span>{countries}</span>
                    <span>{director}</span>
                    {imdb}
                    {tomato}
                    {popcorn}
                </div>
                <div class="overview preview">{overview}</div>
                <div class="tags">{tags}</div>
            </div>
            <div class="score-box">
                <div class="score-number">{score}</div>
                <div class="score-label" style="background:{color};">Matching</div>
            </div>
        </summary>
        <div class="expanded-overview">{overview}</div>
    </details>
    """


HTML = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Movie Finder</title>
    <style>
    * { box-sizing: border-box; }
    html, body {
        margin: 0;
        min-height: 100%;
        background: #ffffff;
        color: #222222;
        font-family: Arial, sans-serif;
    }
    .page {
        width: min(900px, 94vw);
        margin: 0 auto;
    }
    .home {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .search {
        width: min(520px, 84vw);
    }
    .results-search {
        padding: 28px 0 22px;
        margin: 0 auto;
    }
    .search-input {
        width: 100%;
        height: 42px;
        border: 0;
        outline: 0;
        border-radius: 8px;
        background: #e9e9e9;
        color: #222222;
        font: 18px Arial, sans-serif;
        padding: 0 14px;
    }
    .search-input::placeholder {
        color: #7f7f7f;
        opacity: 1;
    }
    .mode-buttons {
        display: flex;
        gap: 8px;
        justify-content: center;
        margin-top: 10px;
    }
    .mode-button {
        border: 0;
        border-radius: 999px;
        padding: 8px 12px;
        background: #eeeeee;
        color: #555555;
        font: 12px Arial, sans-serif;
        cursor: pointer;
    }
    .mode-button.active {
        background: #d7e7ff;
        color: #222222;
        font-weight: 700;
    }
    .hidden-submit {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    }
    .movie-card {
        display: block;
        background: #f3f3f3;
        border: 1px solid #e4e4e4;
        border-radius: 8px;
        margin: 0 0 15px;
        overflow: hidden;
    }
    .movie-card summary {
        list-style: none;
        display: grid;
        grid-template-columns: 92px minmax(0, 1fr) 92px;
        gap: 18px;
        padding: 16px;
        cursor: pointer;
        align-items: stretch;
    }
    .movie-card summary::-webkit-details-marker { display: none; }
    .poster {
        width: 92px;
        height: 138px;
        border-radius: 4px;
        object-fit: cover;
        display: block;
        background: #dddddd;
    }
    .poster-fallback {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #777777;
        font-size: 12px;
        text-align: center;
    }
    .movie-main {
        min-width: 0;
        display: flex;
        flex-direction: column;
    }
    .movie-title {
        font-size: 23px;
        line-height: 1.15;
        font-weight: 700;
        margin: 0 0 7px;
        color: #111111;
    }
    .movie-facts {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 7px 12px;
        font-size: 15px;
        line-height: 1.25;
        color: #222222;
        margin-bottom: 9px;
    }
    .rating-link {
        color: #222222;
        text-decoration: none;
        font-weight: 500;
    }
    .rating-link:hover { text-decoration: underline; }
    .star-icon {
        color: #f4b400;
        font-size: 16px;
        line-height: 1;
    }
    .overview {
        font-size: 15px;
        line-height: 1.38;
        color: #333333;
    }
    .preview {
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .tags {
        margin-top: auto;
        padding-top: 11px;
        max-height: 28px;
        overflow: hidden;
    }
    .tag {
        display: inline-block;
        margin: 0 6px 6px 0;
        padding: 4px 8px;
        border-radius: 999px;
        background: #ffffff;
        color: #555555;
        font-size: 12px;
        line-height: 1;
    }
    .score-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-width: 88px;
    }
    .score-number {
        font-size: 42px;
        line-height: 1;
        font-weight: 700;
        color: #111111;
        margin-bottom: 10px;
    }
    .score-label {
        width: 84px;
        border-radius: 7px;
        padding: 7px 0;
        text-align: center;
        font-size: 13px;
        font-weight: 700;
        color: #222222;
    }
    .expanded-overview {
        border-top: 1px solid #e0e0e0;
        padding: 0 16px 16px 126px;
        font-size: 15px;
        line-height: 1.5;
        color: #333333;
    }
    .load-more {
        display: flex;
        justify-content: center;
        padding: 4px 0 32px;
    }
    .plus {
        width: 34px;
        height: 34px;
        border-radius: 999px;
        border: 1px solid #d4d4d4;
        background: #f4f4f4;
        color: #555555;
        font: 22px/31px Arial, sans-serif;
        text-align: center;
        text-decoration: none;
    }
    @media (max-width: 720px) {
        .movie-card summary {
            grid-template-columns: 76px minmax(0, 1fr);
            gap: 12px;
        }
        .poster {
            width: 76px;
            height: 114px;
        }
        .score-box {
            grid-column: 1 / -1;
            flex-direction: row;
            justify-content: flex-end;
            gap: 12px;
        }
        .score-number {
            font-size: 32px;
            margin-bottom: 0;
        }
        .expanded-overview {
            padding: 0 14px 14px;
        }
        .movie-title {
            font-size: 19px;
        }
    }
    </style>
</head>
<body>
    {% if not query %}
    <main class="home">
        <form class="search" method="get">
            <input class="search-input" name="q" placeholder="Type what you want" autofocus onkeydown="if(event.key==='Enter'){event.preventDefault(); this.form.requestSubmit();}">
            <input type="hidden" name="mode" value="{{ mode }}">
            <button class="hidden-submit" type="submit">Search</button>
            <div class="mode-buttons">
                <button class="mode-button {% if mode == 'specific' %}active{% endif %}" type="submit" onclick="this.form.mode.value='specific'">Looking for specific film</button>
                <button class="mode-button {% if mode == 'recommendation' %}active{% endif %}" type="submit" onclick="this.form.mode.value='recommendation'">Looking for recommendation</button>
            </div>
        </form>
    </main>
    {% else %}
    <main class="page">
        <form class="search results-search" method="get">
            <input class="search-input" name="q" value="{{ query }}" placeholder="Type what you want" onkeydown="if(event.key==='Enter'){event.preventDefault(); this.form.requestSubmit();}">
            <input type="hidden" name="mode" value="{{ mode }}">
            <button class="hidden-submit" type="submit">Search</button>
            <div class="mode-buttons">
                <button class="mode-button {% if mode == 'specific' %}active{% endif %}" type="submit" onclick="this.form.mode.value='specific'">Looking for specific film</button>
                <button class="mode-button {% if mode == 'recommendation' %}active{% endif %}" type="submit" onclick="this.form.mode.value='recommendation'">Looking for recommendation</button>
            </div>
        </form>
        <section class="results">
            {{ cards|safe }}
        </section>
        {% if has_more %}
        <div class="load-more">
            <a class="plus" href="{{ more_url }}">+</a>
        </div>
        {% endif %}
    </main>
    {% endif %}
</body>
</html>
"""


@app.route("/")
def index():
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "recommendation").strip()
    mode = "specific" if mode == "specific" else "recommendation"
    visible = request.args.get("n", "5")
    try:
        visible = max(5, min(int(visible), 30))
    except ValueError:
        visible = 5

    cards = ""
    has_more = False
    more_url = ""
    if query:
        results = recommend_movies(query, top_n=30, mode=mode)
        cards = "".join(movie_card(row) for _, row in results.head(visible).iterrows())
        has_more = visible < len(results)
        more_url = url_for("index", q=query, mode=mode, n=visible + 5)

    return render_template_string(
        HTML,
        query=query,
        mode=mode,
        cards=cards,
        has_more=has_more,
        more_url=more_url,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8510, debug=False)
