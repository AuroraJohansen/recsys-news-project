from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    df["topics_text"] = df["topics"].apply(_safe_join_list)
    df["category_text"] = df["category_str"].fillna("")
    df["entities_text"] = df["entity_groups"].apply(_safe_join_list)

    df["article_text"] = (
        df["title_text"] + " "
        + df["subtitle_text"] + " "
        + df["topics_text"] + " "
        + df["category_text"] + " "
        + df["entities_text"]
    ).str.strip()

    return df[["article_id", "article_text"]]


def fit_vectorizer(article_text_df: pd.DataFrame, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    article_matrix = vectorizer.fit_transform(article_text_df["article_text"])
    return vectorizer, article_matrix


def build_article_embeddings(
    article_text_df: pd.DataFrame,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 256,
    cache_path: str | None = None,
):
    """Encode articles with a multilingual sentence-transformer model.

    Returns a (n_articles, embedding_dim) float32 numpy array.
    If cache_path is given, loads from disk on subsequent runs.
    """
    from sentence_transformers import SentenceTransformer

    if cache_path is not None:
        import os
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)

    model = SentenceTransformer(model_name)
    texts = article_text_df["article_text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if cache_path is not None:
        np.save(cache_path, embeddings)
        print(f"Embeddings saved to {cache_path}")

    return embeddings


def build_article_id_to_index(article_text_df: pd.DataFrame):
    return {
        article_id: idx
        for idx, article_id in enumerate(article_text_df["article_id"])
    }


def build_user_profile(
    user_id,
    history: pd.DataFrame,
    article_matrix,
    article_id_to_idx,
    decay_days: float = 30,
):
    """Build a weighted user profile vector from reading history.

    Weights each article by recency (exponential decay) and engagement
    (scroll percentage), then returns a normalized weighted sum.
    """
    user_history = history[history["user_id"] == user_id]

    rows = []
    for _, row in user_history.iterrows():
        article_ids = row["article_id_fixed"]
        if article_ids is None:
            continue
        scrolls = row["scroll_percentage_fixed"]
        times = row["impression_time_fixed"]
        for i, article_id in enumerate(article_ids):
            scroll = scrolls[i] if (scrolls is not None and i < len(scrolls)) else None
            if hasattr(times, "__len__") and not isinstance(times, str):
                time = times[i] if i < len(times) else times[-1]
            else:
                time = times
            rows.append({"article_id": article_id, "time": time, "scroll": scroll})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df[df["article_id"].isin(article_id_to_idx)]
    if df.empty:
        return None

    latest = df["time"].max()
    df["age_days"] = (latest - df["time"]).dt.total_seconds() / 86400
    df["recency_weight"] = np.exp(-df["age_days"] / decay_days)

    df["scroll"] = pd.to_numeric(df["scroll"], errors="coerce").clip(0, 1).fillna(0.5)
    df["weight"] = df["recency_weight"] * (0.5 + 0.5 * df["scroll"])
    df["weight"] /= df["weight"].sum()

    indices = [article_id_to_idx[a] for a in df["article_id"]]
    user_vector = np.asarray(df["weight"].values @ article_matrix[indices])

    return user_vector


def rank_candidates(user_vector, candidate_article_ids, article_matrix, article_id_to_idx):
    valid_candidates = [a for a in candidate_article_ids if a in article_id_to_idx]

    if not valid_candidates:
        return []

    candidate_indices = [article_id_to_idx[a] for a in valid_candidates]
    candidate_matrix = article_matrix[candidate_indices]
    user_vector = np.asarray(user_vector).reshape(1, -1)
    scores = cosine_similarity(user_vector, candidate_matrix).flatten()

    ranked = sorted(zip(valid_candidates, scores), key=lambda x: x[1], reverse=True)
    return [article_id for article_id, _ in ranked]


class ContentBasedRecommender:
    """Content-based recommender using multilingual sentence embeddings.

    Follows the shared fit/rank interface used by all models in this project.
    Embeddings are built at init time and optionally cached to disk.
    """

    def __init__(
        self,
        articles_df: pd.DataFrame,
        history_df: pd.DataFrame,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        cache_path: str | None = None,
        decay_days: float = 30,
    ):
        article_text_df = build_article_text(articles_df)
        self.article_id_to_idx = build_article_id_to_index(article_text_df)
        self.article_matrix = build_article_embeddings(
            article_text_df, model_name=model_name, cache_path=cache_path
        )
        self.history_df = history_df
        self.decay_days = decay_days

    def fit(self, train_behaviors):
        return self  # embeddings built at init; nothing to train

    def rank(self, candidates, context=None):
        if context is None:
            return list(candidates)

        user_vector = build_user_profile(
            context.user_id,
            self.history_df,
            self.article_matrix,
            self.article_id_to_idx,
            self.decay_days,
        )

        if user_vector is None:
            return list(candidates)  # cold-start: no history, return as-is

        return rank_candidates(
            user_vector, candidates, self.article_matrix, self.article_id_to_idx
        )