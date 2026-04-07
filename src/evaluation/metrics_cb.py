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