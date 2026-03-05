import math

def precision_at_k(ranked, relevant_set, k):
    topk = ranked[:k]
    hits = sum(1 for a in topk if a in relevant_set)
    return hits / k

