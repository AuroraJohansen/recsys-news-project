import math


def precision_at_k(ranked, relevant_set, k):
    topk = ranked[:k]
    hits = sum(1 for a in topk if a in relevant_set)
    return hits / k


def recall_at_k(ranked, relevant_set, k):
    if len(relevant_set) == 0:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for a in topk if a in relevant_set)
    return hits / len(relevant_set)


def mrr(ranked, relevant_set):
    for i, a in enumerate(ranked):
        if a in relevant_set:
            return 1 / (i + 1)
    return 0.0


def ndcg_at_k(ranked, relevant_set, k):
    dcg = sum(
        1 / math.log2(i + 2)
        for i, a in enumerate(ranked[:k]) if a in relevant_set
    )
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant_set), k)))
    return dcg / ideal if ideal > 0 else 0.0
