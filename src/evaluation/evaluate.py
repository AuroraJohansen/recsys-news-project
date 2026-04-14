from .metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k, novelty_at_k, redundancy_at_k, coverage
import time

def evaluate_on_behaviors(model, behaviors_df, ks=(1, 3, 5, 10),
    item_popularity=None, catalog_size=None, max_rows=None, warmup=0):
    precision_scores = {k: [] for k in ks}
    recall_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    novelty_scores = {k: [] for k in ks}
    redundancy_scores = {k: [] for k in ks}
    
    mrr_scores = []
    latencies = []
    all_recommendations = []

    it = behaviors_df.itertuples(index=False)
    if max_rows is not None:
        import itertools
        it = itertools.islice(it, max_rows)

    for i, row in enumerate(it):
        candidates = getattr(row, "article_ids_inview")
        clicked = getattr(row, "article_ids_clicked")
        relevant = set(clicked) if clicked is not None else set()

        start = time.perf_counter()
        ranked = model.rank(candidates, context=row)
        end = time.perf_counter()

        if i >= warmup:
            latencies.append(end - start)

        all_recommendations.append(ranked)

        mrr_scores.append(mrr(ranked, relevant))

        for k in ks:
            precision_scores[k].append(precision_at_k(ranked, relevant, k))
            recall_scores[k].append(recall_at_k(ranked, relevant, k))
            ndcg_scores[k].append(ndcg_at_k(ranked, relevant, k))

            if item_popularity:
                novelty_scores[k].append(
                    novelty_at_k(ranked, item_popularity, k)
                )

            redundancy_scores[k].append(
                redundancy_at_k(ranked, k)
            )

    n = len(mrr_scores)
    results = {
        "n_events": n,
        "mrr": sum(mrr_scores) / n if n else 0.0,
    }

    for k in ks:
        results[f"precision@{k}"] = sum(precision_scores[k]) / n if n else 0.0
        results[f"recall@{k}"] = sum(recall_scores[k]) / n if n else 0.0
        results[f"ndcg@{k}"] = sum(ndcg_scores[k]) / n if n else 0.0

        if item_popularity:
            results[f"novelty@{k}"] = sum(novelty_scores[k]) / n if n else 0.0

        results[f"redundancy@{k}"] = sum(redundancy_scores[k]) / n if n else 0.0

    if catalog_size:
        results["coverage"] = coverage(all_recommendations, catalog_size)

    if len(latencies) > 0:
        total_time = sum(latencies)

        results["avg_latency"] = total_time / len(latencies)
        results["p95_latency"] = sorted(latencies)[int(0.95 * len(latencies))]
        results["throughput"] = len(latencies) / total_time  # events per second

    return results
