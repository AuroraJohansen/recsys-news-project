from .metrics_cb import precision_at_k, recall_at_k
from src.models.content_based import build_user_profile, rank_candidates


def evaluate_on_behaviors(
    behaviors_df,
    history_df,
    article_matrix,
    article_id_to_idx,
    k=10,
    max_rows=None,
):
    precisions = []
    recalls = []

    it = behaviors_df.itertuples(index=False)

    if max_rows is not None:
        import itertools
        it = itertools.islice(it, max_rows)

    for row in it:
        user_id = getattr(row, "user_id")
        candidates = getattr(row, "article_ids_inview")
        clicked = getattr(row, "article_ids_clicked")
        relevant = set(clicked)

        user_vector = build_user_profile(
            user_id,
            history_df,
            article_matrix,
            article_id_to_idx
        )

        if user_vector is None:
            continue

        ranked = rank_candidates(
            user_vector,
            candidates,
            article_matrix,
            article_id_to_idx
        )

        precisions.append(precision_at_k(ranked, relevant, k))
        recalls.append(recall_at_k(ranked, relevant, k))

    return {
        f"precision@{k}": sum(precisions) / len(precisions) if precisions else 0.0,
        f"recall@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        "n_events": len(precisions),
    }