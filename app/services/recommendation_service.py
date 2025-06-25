import random
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale

# Constants
INTERACTION_WEIGHTS = {
    "coldStart": {"like": 1.0, "neutral": 0.5, "dislike": 0.0},
    "like": 1.0,
    "bookmark": 0.5,
    "view": 0.2,
}


class RecommendationEngine:
    def __init__(self, recipes_df: pd.DataFrame, interactions_df: pd.DataFrame):
        self.recipes_df = recipes_df
        self.interactions_df = interactions_df

        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.recipe_map = None
        self.recipe_sim_matrix = None
        self.user_map = None
        self.used_mock = False

    def preprocess(self):
        # Content-based filtering
        tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = tfidf.fit_transform(self.recipes_df["combined_text"])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(
            self.recipes_df.index, index=self.recipes_df["recipe_id"]
        ).drop_duplicates()

        # Collaborative filtering
        self.interactions_df["score"] = self.interactions_df.apply(
            lambda row: INTERACTION_WEIGHTS[row["type"]].get(row["value"], 0.0)
            if row["type"] == "coldStart"
            else INTERACTION_WEIGHTS.get(row["type"], 0.0),
            axis=1,
        )

        self.user_map = {
            uid: i for i, uid in enumerate(self.interactions_df["user_id"].unique())
        }
        self.recipe_map = {
            rid: i for i, rid in enumerate(self.interactions_df["recipe_id"].unique())
        }
        self.interactions_df["user_idx"] = self.interactions_df["user_id"].map(
            self.user_map
        )
        self.interactions_df["recipe_idx"] = self.interactions_df["recipe_id"].map(
            self.recipe_map
        )

        interaction_matrix = csr_matrix(
            (
                self.interactions_df["score"],
                (self.interactions_df["user_idx"], self.interactions_df["recipe_idx"]),
            ),
            shape=(len(self.user_map), len(self.recipe_map)),
        )

        self.recipe_sim_matrix = cosine_similarity(interaction_matrix.T)

    def adaptive_weights(self, recipe_id: str) -> Tuple[float, float]:
        """Adjust weights based on recipe interaction density."""
        count = self.interactions_df[
            self.interactions_df["recipe_id"] == str(recipe_id)
        ].shape[0]
        total = len(self.interactions_df)
        ratio = count / total if total > 0 else 0
        cf_weight = min(0.8, ratio + 0.1)
        cbf_weight = 1.0 - cf_weight
        return cbf_weight, cf_weight

    def get_hybrid_recommendations(
        self, recipe_id: str, cbf_weight=0.6, cf_weight=0.4, top_k=100, sample_n=20
    ) -> pd.DataFrame:
        try:
            cbf_idx = self.indices[recipe_id]
        except KeyError:
            return pd.DataFrame(
                columns=["recipe_id", "title", "image", "author_id", "combined_text"]
            )

        cbf_scores = self.cosine_sim[cbf_idx]
        cf_scores_full = np.zeros_like(cbf_scores)

        cf_idx = self.recipe_map.get(str(recipe_id))
        if cf_idx is not None:
            cf_scores_partial = self.recipe_sim_matrix[cf_idx]
            for rid_str, partial_idx in self.recipe_map.items():
                full_idx = self.indices.get(rid_str)
                if full_idx is not None:
                    cf_scores_full[full_idx] = cf_scores_partial[partial_idx]

        cbf_scores_norm = minmax_scale(cbf_scores)
        cf_scores_norm = minmax_scale(cf_scores_full)
        hybrid_scores = cbf_weight * cbf_scores_norm + cf_weight * cf_scores_norm

        sim_scores = sorted(
            [(i, score) for i, score in enumerate(hybrid_scores) if i != cbf_idx],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        sampled_indices = random.sample(sim_scores, min(sample_n, len(sim_scores)))
        hybrid_indices = [i for i, _ in sampled_indices]

        return self.recipes_df.iloc[hybrid_indices][
            ["recipe_id", "title", "image", "author_id", "combined_text"]
        ]

    def get_trending_recipes(self, n=20) -> pd.DataFrame:
        top_recipe_ids = (
            self.interactions_df[self.interactions_df["type"] == "like"]
            .groupby("recipe_id")
            .size()
            .sort_values(ascending=False)
            .head(n)
            .index
        )
        return self.recipes_df[self.recipes_df["recipe_id"].isin(top_recipe_ids)][
            ["recipe_id", "title"]
        ]
