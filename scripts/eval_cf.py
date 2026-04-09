import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
from src.data.load_data import load_behaviors, load_history
from src.models.collaborative import ItemBasedCFRecommender
from src.evaluation.evaluate import evaluate_on_behaviors


DATA_ROOT = Path("data/raw/ebnerd_large")
OUT_DIR = Path("reports/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    train_beh = load_behaviors(DATA_ROOT, "train")
    val_beh = load_behaviors(DATA_ROOT, "validation")

    # load history
    history = load_history(DATA_ROOT, "train")

    # user_history
    user_history = dict(zip(
        history["user_id"],
        history["article_id_fixed"]
    ))

    results = {}

    # CF model
    cf = ItemBasedCFRecommender(user_history).fit()
    results["item_cf"] = evaluate_on_behaviors(cf, val_beh, max_rows=10000)

    print(results)

    (OUT_DIR / "cf.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()