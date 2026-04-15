from src.models.content_based_tf_idf_2 import build_user_profile, rank_candidates
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


class ContentBasedRecommender:
    def __init__(self, history_df, article_matrix, article_id_to_idx):
        self.history = history_df
        self.article_matrix = article_matrix
        self.article_id_to_idx = article_id_to_idx

    def score_candidates(self, candidates, context=None):
        user_id = getattr(context, "user_id")

        user_vector = build_user_profile(
            user_id,
            self.history,
            self.article_matrix,
            self.article_id_to_idx
        )

        if user_vector is None:
            return {}

        valid_candidates = [
            aid for aid in candidates if aid in self.article_id_to_idx]

        if not valid_candidates:
            return {}

        candidate_indices = [self.article_id_to_idx[aid]
                             for aid in valid_candidates]
        candidate_matrix = self.article_matrix[candidate_indices]

        user_vector = np.asarray(user_vector)
        scores = cosine_similarity(user_vector, candidate_matrix).flatten()

        return {
            aid: float(score)
            for aid, score in zip(valid_candidates, scores)
        }

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
