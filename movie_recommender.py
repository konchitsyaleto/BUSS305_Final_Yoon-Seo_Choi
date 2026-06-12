from pathlib import Path
from functools import lru_cache
import re

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

GENERIC_QUERY_TERMS = {
    "movie",
    "movies",
    "film",
    "films",
    "watch",
    "watching",
    "see",
    "easily",
    "easy",
    "good",
    "great",
    "best",
    "recommend",
    "recommendation",
    "looking",
    "want",
    "something",
}

COUNTRY_EXPANSIONS = {
    "american": "united states america usa",
    "usa": "united states america",
    "us": "united states america",
    "british": "united kingdom britain england",
    "english": "united kingdom britain england",
    "korean": "south korea korea",
    "japanese": "japan japanese",
    "french": "france french",
    "italian": "italy italian",
    "chinese": "china chinese hong kong taiwan",
    "hongkong": "hong kong",
    "indian": "india hindi bollywood",
}

COUNTRY_TARGETS = {
    "american": ["united states of america", "united states", "usa"],
    "usa": ["united states of america", "united states", "usa"],
    "us": ["united states of america", "united states", "usa"],
    "british": ["united kingdom", "england", "britain"],
    "english": ["united kingdom", "england", "britain"],
    "korean": ["south korea", "korea"],
    "japanese": ["japan"],
    "french": ["france"],
    "italian": ["italy"],
    "chinese": ["china", "hong kong", "taiwan"],
    "hongkong": ["hong kong"],
    "indian": ["india"],
}

GENRE_TARGETS = {
    "romance": ["romance"],
    "romantic": ["romance"],
    "comedy": ["comedy"],
    "funny": ["comedy"],
    "horror": ["horror"],
    "scary": ["horror"],
    "thriller": ["thriller", "mystery", "suspense"],
    "suspense": ["thriller", "mystery", "suspense"],
    "mystery": ["mystery"],
    "action": ["action"],
    "adventure": ["adventure"],
    "drama": ["drama"],
    "family": ["family"],
    "animation": ["animation"],
    "animated": ["animation"],
    "documentary": ["documentary"],
    "fantasy": ["fantasy"],
    "sci": ["science fiction"],
    "science": ["science fiction"],
}

CLASSIC_TERMS = {"classic", "classical", "old", "older", "vintage"}


@lru_cache(maxsize=1)
def load_artifacts():
    specific_vectorizer = joblib.load(ARTIFACT_DIR / "tfidf_vectorizer.pkl")
    specific_tfidf_matrix = sparse.load_npz(ARTIFACT_DIR / "movie_tfidf_matrix.npz")
    recommendation_vectorizer = joblib.load(
        ARTIFACT_DIR / "recommendation_tfidf_vectorizer.pkl"
    )
    recommendation_tfidf_matrix = sparse.load_npz(
        ARTIFACT_DIR / "movie_recommendation_tfidf_matrix.npz"
    )
    metadata_path = ARTIFACT_DIR / "movie_tfidf_metadata.csv"
    compressed_metadata_path = ARTIFACT_DIR / "movie_tfidf_metadata.csv.gz"
    if compressed_metadata_path.exists():
        movie_metadata = pd.read_csv(compressed_metadata_path)
    else:
        movie_metadata = pd.read_csv(metadata_path)
    return (
        specific_vectorizer,
        specific_tfidf_matrix,
        recommendation_vectorizer,
        recommendation_tfidf_matrix,
        movie_metadata,
    )


def _clean_display_value(value, fallback=""):
    if pd.isna(value):
        return fallback
    return value


def _extract_match_keywords(query_vector, movie_vector, vectorizer, top_k=8):
    shared_scores = query_vector.multiply(movie_vector).toarray().flatten()
    if shared_scores.max() <= 0:
        return []

    feature_names = vectorizer.get_feature_names_out()
    top_indices = np.argsort(shared_scores)[::-1]
    keywords = []
    for idx in top_indices:
        if shared_scores[idx] <= 0:
            break
        word = feature_names[idx]
        if len(word) < 3 or word.isdigit():
            continue
        keywords.append(word)
        if len(keywords) >= top_k:
            break
    return keywords


def _query_term_coverage(processed_query, vectorizer):
    tokens = _normalize_query_text(processed_query).split()
    if not tokens:
        return 0.0

    vocabulary = vectorizer.vocabulary_
    matched_tokens = sum(1 for token in tokens if token in vocabulary)
    return matched_tokens / len(tokens)


def _calibrated_match_scores(
    similarity_scores,
    vote_quality,
    completeness,
    eligible_for_recommendation,
    country_match,
    genre_match,
    has_genre_constraint,
    classic_match,
    has_classic_constraint,
    term_coverage,
    mode,
    exact_title,
    partial_title,
):
    quality = np.clip(vote_quality * 0.70 + completeness * 0.30, 0, 1)
    text_strength = np.clip(similarity_scores / 0.70, 0, 1) * term_coverage

    if mode == "specific":
        title_strength = exact_title.astype(float) + partial_title.astype(float) * 0.65
        evidence = np.maximum(text_strength, title_strength)
        scores = 100 * evidence * (0.90 + quality * 0.10)
        return np.where(exact_title, 100, np.clip(scores, 0, 100))

    structured_parts = []
    if country_match.max() > 0:
        structured_parts.append(country_match)
    if has_genre_constraint:
        structured_parts.append(genre_match)
    if has_classic_constraint:
        structured_parts.append(np.clip(classic_match, 0, 1))

    if structured_parts:
        structured_strength = np.mean(np.vstack(structured_parts), axis=0)
        structured_weight = min(0.70, 0.35 + len(structured_parts) * 0.12)
        evidence = (
            text_strength * (1 - structured_weight)
            + structured_strength * structured_weight
        )
    else:
        evidence = text_strength

    scores = 100 * evidence * (0.80 + quality * 0.20)
    scores *= np.where(eligible_for_recommendation, 1.0, 0.45)
    return np.clip(scores, 0, 100)


def _normalize_query_text(value):
    text = "" if value is None else str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _prepare_query_for_mode(query, mode):
    normalized = _normalize_query_text(query)
    tokens = normalized.split()

    if mode == "specific":
        return query

    filtered_tokens = [
        token for token in tokens if token not in GENERIC_QUERY_TERMS
    ]
    expanded_tokens = []
    for token in filtered_tokens:
        expansion = COUNTRY_EXPANSIONS.get(token)
        if expansion:
            continue
        expanded_tokens.append(token)

    return " ".join(expanded_tokens) if expanded_tokens else query


def _country_match_scores(query, movie_metadata):
    tokens = _normalize_query_text(query).split()
    targets = []
    for token in tokens:
        targets.extend(COUNTRY_TARGETS.get(token, []))
    if not targets:
        return np.zeros(len(movie_metadata))

    countries = (
        movie_metadata["display_countries"]
        if "display_countries" in movie_metadata.columns
        else movie_metadata["tmdb_production_countries"]
    )
    countries = countries.fillna("").astype(str).str.lower()
    score = np.zeros(len(movie_metadata))
    for target in targets:
        score += countries.str.contains(target, regex=False, na=False).to_numpy()
    return np.clip(score, 0, 1)


def _genre_match_scores(query, movie_metadata):
    tokens = _normalize_query_text(query).split()
    targets = []
    for token in tokens:
        targets.extend(GENRE_TARGETS.get(token, []))
    if not targets:
        return np.zeros(len(movie_metadata)), False

    genres = movie_metadata["tmdb_genres"].fillna("").astype(str).str.lower()
    score = np.zeros(len(movie_metadata))
    for target in set(targets):
        score += genres.str.contains(target, regex=False, na=False).to_numpy()
    return np.clip(score, 0, 1), True


def _classic_match_scores(query, movie_metadata):
    tokens = set(_normalize_query_text(query).split())
    if not tokens.intersection(CLASSIC_TERMS):
        return np.ones(len(movie_metadata)), False

    years = pd.to_numeric(movie_metadata["release_year"], errors="coerce")
    classic = (years.notna() & (years <= 2000)).to_numpy()
    older_bonus = np.zeros(len(movie_metadata))
    older_bonus += (years.fillna(9999).to_numpy() <= 1970).astype(float) * 0.25
    older_bonus += (years.fillna(9999).to_numpy() <= 1950).astype(float) * 0.25
    return classic.astype(float) + older_bonus, True


def _quality_scores(movie_metadata):
    vote_count = pd.to_numeric(
        movie_metadata["tmdb_vote_count"], errors="coerce"
    ).fillna(0)
    vote_quality = np.clip(np.log1p(vote_count).to_numpy() / 10.0, 0, 1)

    has_overview = movie_metadata["tmdb_overview"].fillna("").astype(str).str.len() > 80
    has_keywords = movie_metadata["tmdb_keywords"].fillna("").astype(str).str.len() > 0
    has_poster = movie_metadata["poster_url"].fillna("").astype(str).str.len() > 0
    has_year = movie_metadata["release_year"].notna()
    has_rating = pd.to_numeric(
        movie_metadata["tmdb_vote_average"], errors="coerce"
    ).fillna(0) > 0

    completeness = (
        has_overview.astype(float) * 0.30
        + has_keywords.astype(float) * 0.20
        + has_poster.astype(float) * 0.20
        + has_year.astype(float) * 0.15
        + has_rating.astype(float) * 0.15
    ).to_numpy()
    eligible_for_recommendation = (
        (vote_count >= 100)
        & has_overview
        & has_poster
        & has_year
        & has_rating
    ).to_numpy()
    return vote_quality, completeness, eligible_for_recommendation


def recommend_movies(query, top_n=10, mode="recommendation"):
    mode = "specific" if mode == "specific" else "recommendation"
    (
        specific_vectorizer,
        specific_tfidf_matrix,
        recommendation_vectorizer,
        recommendation_tfidf_matrix,
        movie_metadata,
    ) = load_artifacts()
    if mode == "specific":
        vectorizer = specific_vectorizer
        movie_tfidf_matrix = specific_tfidf_matrix
    else:
        vectorizer = recommendation_vectorizer
        movie_tfidf_matrix = recommendation_tfidf_matrix

    processed_query = _prepare_query_for_mode(query, mode)
    processed_data = vectorizer.transform([processed_query])
    similarity_scores = cosine_similarity(processed_data, movie_tfidf_matrix).flatten()
    term_coverage = _query_term_coverage(processed_query, vectorizer)
    vote_quality, completeness, eligible_for_recommendation = _quality_scores(
        movie_metadata
    )
    country_match = _country_match_scores(query, movie_metadata)
    genre_match, has_genre_constraint = _genre_match_scores(query, movie_metadata)
    classic_match, has_classic_constraint = _classic_match_scores(query, movie_metadata)

    normalized_query = _normalize_query_text(query)
    exact_title = np.zeros(len(movie_metadata), dtype=bool)
    partial_title = np.zeros(len(movie_metadata), dtype=bool)
    if normalized_query:
        query_token_count = len(normalized_query.split())
        normalized_titles = movie_metadata["normalized_title"].fillna("")
        exact_title = normalized_titles.eq(normalized_query).to_numpy()
        if mode == "specific" and query_token_count <= 6:
            query_in_title = normalized_titles.str.contains(
                normalized_query, regex=False, na=False
            ).to_numpy()
            title_in_query = normalized_titles.map(
                lambda title: bool(title) and len(title.split()) >= 2 and title in normalized_query
            ).to_numpy()
            partial_title = (query_in_title | title_in_query) & ~exact_title

    if mode == "specific":
        ranking_scores = (
            similarity_scores * (0.60 + 0.40 * vote_quality)
            + vote_quality * 0.015
            + completeness * 0.010
        )
    else:
        ranking_scores = (
            similarity_scores * (0.18 + 0.82 * vote_quality)
            + vote_quality * 0.030
            + completeness * 0.020
            + country_match * 0.050
            + genre_match * 0.075
            + np.clip(classic_match, 0, 1.5) * 0.040
        )
        if country_match.max() > 0:
            ranking_scores *= np.where(country_match > 0, 1.0, 0.20)
        if has_genre_constraint:
            ranking_scores *= np.where(genre_match > 0, 1.0, 0.08)
        if has_classic_constraint:
            ranking_scores *= np.where(classic_match > 0, 1.0, 0.05)
        ranking_scores *= np.where(eligible_for_recommendation, 1.0, 0.08)

    if normalized_query:
        ranking_scores += exact_title.astype(float) * 2.0
        ranking_scores += partial_title.astype(float) * 0.30

    match_score_values = _calibrated_match_scores(
        similarity_scores=similarity_scores,
        vote_quality=vote_quality,
        completeness=completeness,
        eligible_for_recommendation=eligible_for_recommendation,
        country_match=country_match,
        genre_match=genre_match,
        has_genre_constraint=has_genre_constraint,
        classic_match=classic_match,
        has_classic_constraint=has_classic_constraint,
        term_coverage=term_coverage,
        mode=mode,
        exact_title=exact_title,
        partial_title=partial_title,
    )
    ranking_scores = match_score_values

    candidate_indices = ranking_scores.argsort()[::-1][: max(top_n * 50, 500)]
    top_indices = []
    seen_titles = set()
    normalized_titles_list = movie_metadata["normalized_title"].fillna("").to_numpy()
    for index in candidate_indices:
        title_key = normalized_titles_list[index] or f"row-{index}"
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        top_indices.append(index)
        if len(top_indices) >= top_n:
            break
    top_indices = np.array(top_indices)
    results_df = movie_metadata.iloc[top_indices].copy()

    results_df["similarity_score"] = similarity_scores[top_indices]
    results_df["ranking_score"] = ranking_scores[top_indices]
    results_df["match_score"] = [
        int(round(value)) for value in match_score_values[top_indices]
    ]
    results_df["rank"] = range(1, len(results_df) + 1)
    results_df["match_keywords"] = [
        _extract_match_keywords(
            processed_data,
            movie_tfidf_matrix[index],
            vectorizer,
            top_k=8,
        )
        for index in top_indices
    ]
    results_df["display_overview"] = results_df["tmdb_overview"].combine_first(
        results_df["movie_info"]
    )
    results_df["display_genres"] = results_df["tmdb_genres"].fillna("")
    results_df["display_countries"] = results_df["tmdb_production_countries"].fillna("")

    display_cols = [
        "rank",
        "movie_title",
        "release_year",
        "display_countries",
        "display_genres",
        "directors",
        "poster_url",
        "imdb_url",
        "rotten_tomatoes_url",
        "tmdb_vote_average",
        "tomatometer_status",
        "tomatometer_rating",
        "audience_rating",
        "review_count",
        "match_score",
        "match_keywords",
        "display_overview",
        "critics_consensus",
    ]
    cleaned_results = results_df[display_cols].copy()
    for column in cleaned_results.columns:
        cleaned_results[column] = cleaned_results[column].map(
            lambda value: value if isinstance(value, list) else _clean_display_value(value)
        )
    return cleaned_results


if __name__ == "__main__":
    new_data = "dark psychological thriller with suspense and mystery"
    predictions = recommend_movies(new_data, top_n=10)
    print(predictions.head())
