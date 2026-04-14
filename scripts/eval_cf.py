import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
from collections import Counter
from src.data.load_data import load_behaviors, load_history, load_articles
from src.models.collaborative import ItemBasedCFRecommender
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
    articles = load_articles(DATA_ROOT)

    # load history
    history = load_history(DATA_ROOT, "train")

    # user_history
    user_history = dict(zip(
        history["user_id"],
        history["article_id_fixed"]
    ))

    item_popularity = compute_item_popularity(train_beh)
    catalog_size = articles["article_id"].nunique()

    model = ItemBasedCFRecommender(user_history).fit()

    print("Evaluating Item-CF...")

    results = {
        "item_cf": evaluate_on_behaviors(
            model,
            val_beh, 
            item_popularity=item_popularity, 
            catalog_size=catalog_size
        )
    }

    print(json.dumps(results, indent=2))

    (OUT_DIR / "cf.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()