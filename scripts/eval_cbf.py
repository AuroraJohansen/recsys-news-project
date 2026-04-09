import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
from src.data.load_data import load_articles, load_history, load_behaviors
from src.models.content_based import (
    build_article_text,
    fit_vectorizer,
    build_article_id_to_index,
)
from src.models.content_based_model import ContentBasedRecommender
from src.evaluation.evaluate import evaluate_on_behaviors


DATA_ROOT = Path("data/raw/ebnerd_large")
OUT_DIR = Path("reports/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    articles = load_articles(DATA_ROOT)
    history = load_history(DATA_ROOT, "train")
    val_beh = load_behaviors(DATA_ROOT, "validation")


    article_text_df = build_article_text(articles)
    vectorizer, article_matrix = fit_vectorizer(article_text_df)
    article_id_to_idx = build_article_id_to_index(article_text_df)

    
    cbf = ContentBasedRecommender(
        history,
        article_matrix,
        article_id_to_idx
    )

    results = {}
    results["content_based"] = evaluate_on_behaviors(
        cbf,
        val_beh,
        max_rows=10000
    )

    print(results)

    (OUT_DIR / "cbf.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()