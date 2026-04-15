from collections import defaultdict, Counter
from itertools import combinations, islice


class ItemBasedCFRecommender:
    def __init__(self, user_history, max_history=50, max_users_for_fit=50000):
        self.user_history = user_history
        self.max_history = max_history
        self.max_users_for_fit = max_users_for_fit
        self.co_counts = defaultdict(Counter)

    def fit(self):
        for i, articles in enumerate(islice(self.user_history["article_ids_clicked"].values, self.max_users_for_fit)):
            articles = list(dict.fromkeys(articles))[:self.max_history]

            for a, b in combinations(articles, 2):
                self.co_counts[a][b] += 1
                self.co_counts[b][a] += 1

        return self
    
    def predict(self, user_id, candidate_item):
        user_rows = self.user_history[self.user_history["user_id"] == user_id]

        history = []
        for arr in user_rows["article_ids_clicked"]:
            if arr is not None and len(arr) > 0:
                history.extend(arr)

        if not history:
            return 0.0

        score = 0.0
        for item in history:
            if item == candidate_item:
                continue
            score += self.co_counts[item].get(candidate_item, 0)

        return score


    def score_candidates(self, candidates, context=None):
        user_id = context.user_id
        return {
            a: float(self.predict(user_id, a))
            for a in candidates
        }

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
 