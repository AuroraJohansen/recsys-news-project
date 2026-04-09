from .metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k


def evaluate_on_behaviors(model, behaviors_df, ks=(1, 3, 5, 10), max_rows=None):
    precision_scores = {k: [] for k in ks}
    recall_scores = {k: [] for k in ks}
    mrr_scores = []
    ndcg_scores = {k: [] for k in ks}

    it = behaviors_df.itertuples(index=False)
    if max_rows is not None:
        import itertools
        it = itertools.islice(it, max_rows)

    for row in it:
        candidates = getattr(row, "article_ids_inview")
        clicked = getattr(row, "article_ids_clicked")
        relevant = set(clicked) if clicked is not None else set()

        ranked = model.rank(candidates, context=row)

        mrr_scores.append(mrr(ranked, relevant))
        for k in ks:
            precision_scores[k].append(precision_at_k(ranked, relevant, k))
            recall_scores[k].append(recall_at_k(ranked, relevant, k))
            ndcg_scores[k].append(ndcg_at_k(ranked, relevant, k))

    n = len(mrr_scores)
    results = {
        "n_events": n,
        "mrr": sum(mrr_scores) / n if n else 0.0,
    }

    for k in ks:
        results[f"precision@{k}"] = sum(precision_scores[k]) / n if n else 0.0
        results[f"recall@{k}"] = sum(recall_scores[k]) / n if n else 0.0
        results[f"ndcg@{k}"] = sum(ndcg_scores[k]) / n if n else 0.0

    return results
