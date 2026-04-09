from src.models.content_based import build_user_profile, rank_candidates


class ContentBasedRecommender:
    def __init__(self, history_df, article_matrix, article_id_to_idx):
        self.history = history_df
        self.article_matrix = article_matrix
        self.article_id_to_idx = article_id_to_idx

    def rank(self, candidates, context=None):
        user_id = getattr(context, "user_id")

        user_vector = build_user_profile(
            user_id,
            self.history,
            self.article_matrix,
            self.article_id_to_idx
        )

        if user_vector is None:
            return []

        return rank_candidates(
            user_vector,
            candidates,
            self.article_matrix,
            self.article_id_to_idx
        )