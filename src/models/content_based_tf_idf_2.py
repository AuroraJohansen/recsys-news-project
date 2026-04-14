from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


"""
Content-based filtering utilities.

This module builds article representations from content features using TF-IDF,
constructs user profiles from reading history, and ranks candidate articles
by cosine similarity.
"""


def _safe_join_list(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(map(str, value))
    return str(value)


def build_article_text(articles: pd.DataFrame) -> pd.DataFrame:
    df = articles.copy()

    df["title_text"] = df["title"].fillna("")
    #df["subtitle_text"] = df["subtitle"].fillna("")
    #df["body_text"] = df["body"].fillna("")
    df["topics_text"] = df["topics"].apply(_safe_join_list)
    df["category_text"] = df["category_str"].fillna("")
    df["entities_text"] = df["entity_groups"].apply(_safe_join_list)

    # Simple feature weighting by repetition
    df["article_text"] = (
        (df["title_text"] + " ") * 3
        #+ (df["subtitle_text"] + " ") * 2
        + (df["topics_text"] + " ") * 3
        + (df["category_text"] + " ") * 2
        + (df["entities_text"] + " ") * 2
        #+ df["body_text"]
    ).str.strip()

    return df[["article_id", "article_text"]]


def fit_vectorizer(article_text_df: pd.DataFrame, max_features: int = 10000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        lowercase=True,
    )
    article_matrix = vectorizer.fit_transform(article_text_df["article_text"])
    return vectorizer, article_matrix


def build_article_id_to_index(article_text_df: pd.DataFrame):
    return {
        article_id: idx
        for idx, article_id in enumerate(article_text_df["article_id"])
    }


def build_user_profile(user_id, history: pd.DataFrame, article_matrix, article_id_to_idx):
    user_history = history[history["user_id"] == user_id]

    article_ids = []
    for arr in user_history["article_id_fixed"]:
        if arr is None:
            continue
        article_ids.extend(list(arr))

    valid_article_ids = [aid for aid in article_ids if aid in article_id_to_idx]

    if not valid_article_ids:
        return None

    article_indices = [article_id_to_idx[aid] for aid in valid_article_ids]
    profile_matrix = article_matrix[article_indices]

    # Later interactions in the history get slightly higher weight
    weights = np.linspace(1.0, 2.0, num=len(article_indices))
    user_vector = np.asarray(profile_matrix.multiply(weights[:, None]).sum(axis=0))
    user_vector = user_vector / weights.sum()

    return user_vector


def rank_candidates(user_vector, candidate_article_ids, article_matrix, article_id_to_idx):
    valid_candidates = [aid for aid in candidate_article_ids if aid in article_id_to_idx]

    if not valid_candidates:
        return []

    candidate_indices = [article_id_to_idx[aid] for aid in valid_candidates]
    candidate_matrix = article_matrix[candidate_indices]

    user_vector = np.asarray(user_vector)
    scores = cosine_similarity(user_vector, candidate_matrix).flatten()

    ranked = sorted(
        zip(valid_candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [article_id for article_id, _ in ranked]