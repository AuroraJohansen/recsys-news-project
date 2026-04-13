from __future__ import annotations
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


"""
Content-based filtering utilities.

Builds text representations of articles and converts them into TF-IDF vectors.
These representations will later be used to compute similarity between articles
and generate recommendations.
"""


def _safe_join_list(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(map(str, value))
    return str(value)


def build_article_text(articles: pd.DataFrame) -> pd.DataFrame:
    df = articles.copy()

    df["title_text"] = df["title"].fillna("")
    df["subtitle_text"] = df["subtitle"].fillna("")
    df["body_text"] = df["body"].fillna("")
    df["topics_text"] = df["topics"].apply(_safe_join_list)
    df["category_text"] = df["category_str"].fillna("")
    df["entities_text"] = df["entity_groups"].apply(_safe_join_list)

    df["article_text"] = (
        df["title_text"] + " "
        + df["subtitle_text"] + " "
        + df["body_text"] + " "
        + df["topics_text"] + " "
        + df["category_text"] + " "
        + df["entities_text"]
    ).str.strip()

    return df[["article_id", "article_text"]]


def fit_vectorizer(article_text_df: pd.DataFrame, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    article_matrix = vectorizer.fit_transform(article_text_df["article_text"])
    return vectorizer, article_matrix

#TF-IDF - Term Frequency-Inverse Document Frequency. 
#idea is to represent each document (in this case, each article) as a vector of numbers that reflect the importance of each word in the document relative to the entire corpus of documents.
# TF (Term Frequency) measures how frequently a term appears in a document. The more times a word appears in an article, the higher its TF score for that article.
# IDF (Inverse Document Frequency) measures how important a term is across the entire corpus.


def build_article_id_to_index(article_text_df: pd.DataFrame):

    return {
        article_id: idx
        for idx, article_id in enumerate(article_text_df["article_id"])
    }


# article_text_df = text per article 
# article_matrix = TF-IDF vector representation of articles (numerical repr)
# article_id_to_index = mapping from article_id to row index in article_matrix


"""Building user profile from history
    Idea: 
    - For each user, we can look at the articles they have interacted with in their history.
    - We can then aggregate the TF-IDF vectors of those articles to create a user profile vector.
    """

def build_user_profile(user_id, history: pd.DataFrame, article_matrix, article_id_to_idx):
    user_history = history[history["user_id"] == user_id]

    article_ids = []
    for arr in user_history["article_id_fixed"]:
        if arr is None:
            continue
        article_ids.extend(list(arr))

    valid_article_ids = [article_id for article_id in article_ids if article_id in article_id_to_idx]

    if not valid_article_ids:
        return None

    article_indices = [article_id_to_idx[article_id] for article_id in valid_article_ids]
    user_vector = article_matrix[article_indices].mean(axis=0)

    return user_vector

"""Ranking candidatete articles for a user/one impression
    Idea:
    - For a given user, we can compute the cosine similarity between their user profile vector and the TF-IDF vectors of candidate articles.
    - We can then rank the candidate articles based on their similarity scores."""


from sklearn.metrics.pairwise import cosine_similarity

def rank_candidates(user_vector, candidate_article_ids, article_matrix, article_id_to_idx):
    valid_candidates = [article_id for article_id in candidate_article_ids if article_id in article_id_to_idx]

    if not valid_candidates:
        return []

    candidate_indices = [article_id_to_idx[article_id] for article_id in valid_candidates]
    candidate_matrix = article_matrix[candidate_indices]
    user_vector = np.asarray(user_vector)
    scores = cosine_similarity(user_vector, candidate_matrix).flatten()

    ranked = sorted(
        zip(valid_candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [article_id for article_id, _ in ranked]