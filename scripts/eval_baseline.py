import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
from src.data.load_data import load_behaviors, load_articles
from src.models.baseline import RandomRecommender, MostPopularRecommender, MostRecentRecommender
from src.evaluation.evaluate import evaluate_on_behaviors

DATA_ROOT = Path("data/raw/ebnerd_large")
OUT_DIR = Path("reports/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    train_beh = load_behaviors(DATA_ROOT, "train")
    val_beh = load_behaviors(DATA_ROOT, "validation")

    results = {}

    # Random
    rnd = RandomRecommender().fit(train_beh)
    results["random"] = evaluate_on_behaviors(rnd, val_beh)

    # Most popular
    pop = MostPopularRecommender().fit(train_beh)
    results["most_popular"] = evaluate_on_behaviors(pop, val_beh)

    # Most recent
    articles = load_articles(DATA_ROOT)
    time_col = "published_time"
    article_time = dict(zip(articles["article_id"], articles[time_col]))

    rec = MostRecentRecommender(article_time).fit()
    results["most_recent"] = evaluate_on_behaviors(rec, val_beh)

    print(results)
    (OUT_DIR / "baselines.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
