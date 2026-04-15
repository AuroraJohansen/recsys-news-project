class HybridRecommender:
    def __init__(self, cb_model, cf_model, baseline_model, w_cb=0.6, w_cf=0.25, w_base=0.15):
        self.cb_model = cb_model
        self.cf_model = cf_model
        self.baseline_model = baseline_model
        self.w_cb = w_cb
        self.w_cf = w_cf
        self.w_base = w_base

    def _normalize(self, score_dict):
        if not score_dict:
            return {}

        values = list(score_dict.values())
        min_v, max_v = min(values), max(values)

        if max_v == min_v:
            return {k: 1.0 for k in score_dict}

        return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}

    def rank(self, candidates, context=None):
        cb_scores = self._normalize(
            self.cb_model.score_candidates(candidates, context)
        )
        cf_scores = self._normalize(
            self.cf_model.score_candidates(candidates, context)
        )
        base_scores = self._normalize(
            self.baseline_model.score_candidates(candidates, context)
        )

        combined_scores = {}
        for aid in candidates:
            combined_scores[aid] = (
                self.w_cb * cb_scores.get(aid, 0.0)
                + self.w_cf * cf_scores.get(aid, 0.0)
                + self.w_base * base_scores.get(aid, 0.0)
            )

        ranked = sorted(combined_scores.items(),
                        key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in ranked]
