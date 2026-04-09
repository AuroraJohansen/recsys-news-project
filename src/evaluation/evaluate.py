from .metrics import precision_at_k, recall_at_k


def evaluate_on_behaviors(model, behaviors_df, ks=(1, 3, 5, 10), max_rows=None):
    precision_scores = {k: [] for k in ks}
    recall_scores = {k: [] for k in ks}

    it = behaviors_df.itertuples(index=False)
    if max_rows is not None:
        import itertools
        it = itertools.islice(it, max_rows)

    for row in it:
        candidates = getattr(row, "article_ids_inview")
        clicked = getattr(row, "article_ids_clicked")
        relevant = set(clicked) if clicked is not None else set()

        ranked = model.rank(candidates, context=row)

        for k in ks:
            precision_scores[k].append(precision_at_k(ranked, relevant, k))
            recall_scores[k].append(recall_at_k(ranked, relevant, k))

    results = {"n_events": len(next(iter(precision_scores.values())))}

    for k in ks:
        results[f"precision@{k}"] = sum(precision_scores[k]) / \
            len(precision_scores[k])
        results[f"recall@{k}"] = sum(recall_scores[k]) / len(recall_scores[k])

    return results
