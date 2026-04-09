import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
from src.data.load_data import load_articles, load_behaviors, load_history
from src.models.content_based import ContentBasedRecommender
from src.evaluation.evaluate import evaluate_on_behaviors


DATA_ROOT = Path("data/raw/ebnerd_large")
OUT_DIR = Path("reports/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    articles = load_articles(DATA_ROOT)
    history = load_history(DATA_ROOT, "train")
    train_beh = load_behaviors(DATA_ROOT, "train")
    val_beh = load_behaviors(DATA_ROOT, "validation")

    model = ContentBasedRecommender(
        articles_df=articles,
        history_df=history,
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        cache_path="data/article_embeddings.npy",
    )
    model.fit(train_beh)

    results = {"content_based": evaluate_on_behaviors(model, val_beh, max_rows=10000)}

    print(results)

    (OUT_DIR / "cb.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
