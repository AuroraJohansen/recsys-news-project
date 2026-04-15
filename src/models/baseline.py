from collections import Counter
import random


class RandomRecommender:
    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, train_behaviors, articles=None):
        random.seed(self.seed)
        return self

    def rank(self, candidates, context=None):
        cand = list(candidates)
        random.shuffle(cand)
        return cand


class MostPopularRecommender:
    def __init__(self):
        self.popularity = None

    def fit(self, train_behaviors, articles=None):
        c = Counter()
        for clicked in train_behaviors["article_ids_clicked"]:
            c.update(clicked)
        self.popularity = c
        return self

    def rank(self, candidates, context=None):
        return sorted(candidates, key=lambda a: self.popularity.get(a, 0), reverse=True)


class MostRecentRecommender:
    def __init__(self, article_time_map):
        self.article_time_map = article_time_map  # {article_id: timestamp}

    def fit(self, train_behaviors=None, articles=None):
        return self

    def score_candidates(self, candidates, context=None):
        return {a: float(self.article_time_map.get(a, -1).timestamp()) for a in candidates}

    def rank(self, candidates, context=None):
        return sorted(candidates, key=lambda a: self.article_time_map.get(a, -1), reverse=True)
