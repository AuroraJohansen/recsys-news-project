from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

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
    results["random"] = evaluate_on_behaviors(rnd, val_beh, k=10)

    # Most popular
    pop = MostPopularRecommender().fit(train_beh)
    results["most_popular"] = evaluate_on_behaviors(pop, val_beh, k=10)

    # Most recent (må ha tid fra articles)
    articles = load_articles(DATA_ROOT)
    # TODO: finn riktig tidskolonne i articles, f.eks. "published_time"
    time_col = "published_time"  # bytt til korrekt!
    article_time = dict(zip(articles["article_id"], articles[time_col]))

    rec = MostRecentRecommender(article_time).fit()
    results["most_recent"] = evaluate_on_behaviors(rec, val_beh, k=10)

    print(results)
    (OUT_DIR / "baselines.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()