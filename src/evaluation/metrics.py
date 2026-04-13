import math

# Accuracy-oriented metrics:
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

# Beyond Accuracy metrics:
def novelty_at_k(ranked, item_popularity, k):
    topk = ranked[:k]
    total = 0.0

    for item in topk:
        p = item_popularity.get(item, 1e-10)
        total += -math.log2(p)

    return total / k

def coverage(all_recommendations, catalog_size):
    unique_items = set()
    for recs in all_recommendations:
        unique_items.update(recs)
    return len(unique_items) / catalog_size

def redundancy_at_k(ranked, k):
    topk = ranked[:k]
    return 1.0 - (len(set(topk)) / len(topk))
