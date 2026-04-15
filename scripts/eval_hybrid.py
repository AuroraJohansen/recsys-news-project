import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
from collections import Counter

from src.data.load_data import load_behaviors, load_articles, load_history
from src.models.hybrid import HybridRecommender
from src.models.baseline import MostRecentRecommender
from src.models.content_based_model import ContentBasedRecommender
from src.models.collaborative import ItemBasedCFRecommender
from src.models.content_based_tf_idf_2 import (
    build_article_text,
    fit_vectorizer,
    build_article_id_to_index
)
from src.evaluation.evaluate import evaluate_on_behaviors

DATA_ROOT = Path("data/raw/ebnerd_large")
OUT_DIR = Path("reports/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_item_popularity(behaviors_df):
    counter = Counter()

    for row in behaviors_df.itertuples(index=False):
        clicked = getattr(row, "article_ids_clicked")

        if clicked is not None and len(clicked) > 0:
            counter.update(clicked)

    total = sum(counter.values())
    return {item: count / total for item, count in counter.items()}


def main():
    train_beh = load_behaviors(DATA_ROOT, "train")
    val_beh = load_behaviors(DATA_ROOT, "validation")
    train_history = load_history(DATA_ROOT, "train")
    articles = load_articles(DATA_ROOT)

    item_popularity = compute_item_popularity(train_beh)
    catalog_size = articles["article_id"].nunique()

    # --- baseline (MostRecent)
    article_time = dict(zip(articles["article_id"], articles["published_time"]))
    recent_model = MostRecentRecommender(article_time).fit()

    # --- content-based
    article_text_df = build_article_text(articles)
    vectorizer, article_matrix = fit_vectorizer(article_text_df)
    article_id_to_idx = build_article_id_to_index(article_text_df)

    cb_model = ContentBasedRecommender(
        history_df=train_history,
        article_matrix=article_matrix,
        article_id_to_idx=article_id_to_idx
    )

    # --- collaborative filtering
    cf_model = ItemBasedCFRecommender(train_beh).fit()

    # --- hybrid
    hybrid_model = HybridRecommender(
        cb_model,
        cf_model,
        recent_model,
        w_cb=0.6,
        w_cf=0.25,
        w_base=0.15
    )

    print("Evaluating hybrid...")

    results = evaluate_on_behaviors(
        hybrid_model,
        val_beh,
        item_popularity=item_popularity,
        catalog_size=catalog_size
    )

    print(json.dumps(results, indent=2))

    (OUT_DIR / "hybrid.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()