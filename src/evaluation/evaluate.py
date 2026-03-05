from .metrics import precision_at_k

def evaluate_on_behaviors(model, behaviors_df, k=10, max_rows=None):
    precisions = []

    it = behaviors_df.itertuples(index=False)
    if max_rows is not None:
        import itertools
        it = itertools.islice(it, max_rows)

    for row in it:
        candidates = getattr(row, "article_ids_inview")
        clicked = getattr(row, "article_ids_clicked")
        relevant = set(clicked)

        ranked = model.rank(candidates)

        precisions.append(precision_at_k(ranked, relevant, k))

    return {
        f"precision@{k}": sum(precisions) / len(precisions),
        "n_events": len(precisions),
    }