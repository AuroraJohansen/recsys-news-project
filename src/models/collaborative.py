from collections import defaultdict, Counter
from itertools import combinations, islice


class ItemBasedCFRecommender:
    def __init__(self, user_history, max_history=50, max_users_for_fit=50000):
        self.user_history = user_history
        self.max_history = max_history
        self.max_users_for_fit = max_users_for_fit
        self.co_counts = defaultdict(Counter)

    def fit(self):
        for i, articles in enumerate(islice(self.user_history.values(), self.max_users_for_fit)):
            articles = list(dict.fromkeys(articles))[:self.max_history]

            for a, b in combinations(articles, 2):
                self.co_counts[a][b] += 1
                self.co_counts[b][a] += 1

        return self

    def rank(self, candidates, context=None):
        user_id = getattr(context, "user_id", None)
        history = self.user_history.get(user_id, [])[:self.max_history]

        scores = {}

        for candidate in candidates:
            score = 0
            for h in history:
                score += self.co_counts[h].get(candidate, 0)
            scores[candidate] = score

        return sorted(candidates, key=lambda x: scores[x], reverse=True)
 