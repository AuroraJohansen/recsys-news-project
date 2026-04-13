import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
from collections import Counter
from src.data.load_data import load_behaviors, load_articles
from src.models.baseline import RandomRecommender, MostPopularRecommender, MostRecentRecommender
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
    item_popularity = compute_item_popularity(train_beh)
    catalog_size = articles["article_id"].nunique()

    results = {}

    # Random
    rnd = RandomRecommender().fit(train_beh)
    results["random"] = evaluate_on_behaviors(rnd, val_beh, item_popularity=item_popularity, catalog_size=catalog_size)

    # Most popular
    pop = MostPopularRecommender().fit(train_beh)
    results["most_popular"] = evaluate_on_behaviors(pop, val_beh, item_popularity=item_popularity, catalog_size=catalog_size)

    # Most recent
    time_col = "published_time"
    article_time = dict(zip(articles["article_id"], articles[time_col]))

    rec = MostRecentRecommender(article_time).fit()
    results["most_recent"] = evaluate_on_behaviors(rec, val_beh, item_popularity=item_popularity, catalog_size=catalog_size)

    print(results)
    (OUT_DIR / "baselines.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
