import random
from datetime import datetime, timedelta, timezone
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale

INTERACTION_WEIGHTS = {
    "coldstart": {"like": 1.0, "neutral": 0.5, "dislike": 0.0},
    "like": 1.0,
    "bookmark": 0.5,
    "view": 0.2,
}

FEED_TYPES = ["recipe", "discussion", "tip", "community"]

ID_COLUMN_MAP = {
    "recipe": "recipe_id",
    "tip": "post_id",
    "discussion": "post_id",
    "community": "community_id",
}


def build_tfidf_and_similarity(df: pd.DataFrame, id_col: str):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df[id_col]).drop_duplicates()

    return {
        "tfidf_matrix": tfidf_matrix,
        "cosine_sim": cosine_sim,
        "indices": indices,
    }


class RecommendationEngine:
    def __init__(
        self,
        recipes_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        tips_df: pd.DataFrame = pd.DataFrame(),
        discussions_df: pd.DataFrame = pd.DataFrame(),
        communities_df: pd.DataFrame = pd.DataFrame(),
    ):
        self.recipes_df = recipes_df
        self.interactions_df = interactions_df
        self.tips_df = tips_df
        self.discussions_df = discussions_df
        self.communities_df = communities_df

        self.similarity_models = {
            "recipe": {},
            "tip": {},
            "discussion": {},
            "community": {},
        }

        self.sim_matrices = {
            "recipe": {},
            "tip": {},
            "discussion": {},
            "community": {},
        }

        self.data_sufficient = False

    def preprocess_content_based(self):
        if not self.recipes_df.empty:
            self.similarity_models["recipe"] = build_tfidf_and_similarity(
                self.recipes_df, "recipe_id"
            )

        if not self.tips_df.empty:
            self.similarity_models["tip"] = build_tfidf_and_similarity(
                self.tips_df, "post_id"
            )

        if not self.discussions_df.empty:
            self.similarity_models["discussion"] = build_tfidf_and_similarity(
                self.discussions_df, "post_id"
            )

        if not self.communities_df.empty:
            self.similarity_models["community"] = build_tfidf_and_similarity(
                self.communities_df, "community_id"
            )

    def preprocess_collaborative_based(self):
        now = datetime.now(timezone.utc)
        seven_days_ago = now - timedelta(days=7)

        def compute_score(row):
            if row["type"] == "coldstart":
                return INTERACTION_WEIGHTS["coldstart"].get(row["value"], 0.0)
            elif row["type"] == "view":
                return None  # handled separately
            else:
                return INTERACTION_WEIGHTS.get(row["type"], 0.0)

        self.interactions_df["score"] = self.interactions_df.apply(
            compute_score, axis=1
        )

        view_scores = self.compute_view_decay_scores(seven_days_ago)

        non_view_df = self.interactions_df[self.interactions_df["type"] != "view"][
            ["user_id", "item_id", "item_type", "type", "score", "created_at"]
        ]

        self.interactions_df = pd.concat([non_view_df, view_scores], ignore_index=True)

        self.sim_matrices = {}
        for content_type in ["recipe", "tip", "discussion", "community"]:
            df = self.interactions_df[self.interactions_df["item_type"] == content_type]

            if df.empty:
                continue

            user_map = {uid: i for i, uid in enumerate(df["user_id"].unique())}
            item_map = {iid: i for i, iid in enumerate(df["item_id"].unique())}

            if len(user_map) == 0 or len(item_map) == 0:
                continue

            df["user_idx"] = df["user_id"].map(user_map)
            df["item_idx"] = df["item_id"].map(item_map)

            interaction_matrix = csr_matrix(
                (
                    df["score"],
                    (df["user_idx"], df["item_idx"]),
                ),
                shape=(len(user_map), len(item_map)),
            )

            sim_matrix = cosine_similarity(interaction_matrix.T)

            self.sim_matrices[content_type] = {
                "matrix": sim_matrix,
                "item_map": item_map,
                "item_idx_to_id": {v: k for k, v in item_map.items()},
            }

    def compute_view_decay_scores(self, since: datetime) -> pd.DataFrame:
        view_df = self.interactions_df[self.interactions_df["type"] == "view"].explode(
            "timestamps"
        )
        view_df["timestamp"] = pd.to_datetime(view_df["timestamps"], utc=True)

        recent_views = view_df[view_df["timestamp"] >= since]

        view_counts = (
            recent_views.groupby(["user_id", "item_id", "item_type"])
            .agg(view_count=("timestamp", "count"), created_at=("timestamp", "max"))
            .reset_index()
        )

        view_counts["score"] = view_counts["view_count"].apply(
            lambda c: min(0.2 * np.log1p(c), 0.6)
        )
        view_counts["type"] = "view"

        return view_counts[
            ["user_id", "item_id", "item_type", "type", "score", "created_at"]
        ]

    def preprocess(self):
        self.preprocess_content_based()
        self.preprocess_collaborative_based()

    def adaptive_weights(self, item_id: str) -> Tuple[float, float]:
        count = self.interactions_df[
            self.interactions_df["item_id"] == str(item_id)
        ].shape[0]
        total = len(self.interactions_df)
        ratio = count / total if total > 0 else 0
        cf_weight = min(0.8, ratio + 0.1)
        cbf_weight = 1.0 - cf_weight
        return cbf_weight, cf_weight

    def get_hybrid_recommendations(
        self,
        item_id: str,
        content_type: str,
        cbf_weight=0.6,
        cf_weight=0.4,
        top_k=100,
        sample_n=40,
    ) -> pd.DataFrame:
        if item_id not in self.similarity_models.get(content_type, {}).get(
            "indices", {}
        ):
            return pd.DataFrame()

        cbf_idx = self.similarity_models[content_type]["indices"][item_id]
        cbf_scores = self.similarity_models[content_type]["cosine_sim"][cbf_idx]
        cbf_scores_norm = minmax_scale(cbf_scores)

        if (
            content_type in self.sim_matrices
            and item_id in self.sim_matrices[content_type]["item_map"]
        ):
            cf_idx = self.sim_matrices[content_type]["item_map"][item_id]
            cf_scores_partial = self.sim_matrices[content_type]["matrix"][cf_idx]

            cf_scores_full = np.zeros_like(cbf_scores)
            for rid_str, partial_idx in self.sim_matrices[content_type][
                "item_map"
            ].items():
                full_idx = self.similarity_models[content_type]["indices"].get(rid_str)
                if full_idx is not None:
                    cf_scores_full[full_idx] = cf_scores_partial[partial_idx]

            cf_scores_norm = minmax_scale(cf_scores_full)
            hybrid_scores = cbf_weight * cbf_scores_norm + cf_weight * cf_scores_norm
        else:
            hybrid_scores = cbf_scores_norm

        sim_scores = sorted(
            [(i, score) for i, score in enumerate(hybrid_scores) if i != cbf_idx],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        sampled_indices = random.sample(sim_scores, min(sample_n, len(sim_scores)))
        hybrid_indices = [i for i, _ in sampled_indices]

        df = (
            self.communities_df
            if content_type == "community"
            else getattr(self, f"{content_type}s_df")
        )

        if content_type == "community" and "name" in df.columns:
            df = df.rename(columns={"name": "title"})

        if "author_id" not in df.columns:
            df = df.copy()
            df["author_id"] = ""

        df["author_id"] = df["author_id"].fillna("").astype(str)

        id_col = {"recipe": "recipe_id", "community": "community_id"}.get(
            content_type, "post_id"
        )

        return df.iloc[hybrid_indices][[id_col, "title", "image", "author_id"]]

    def get_trending_items(self, content_type: str, n: int = 20) -> pd.DataFrame:
        df_map = {
            "recipe": self.recipes_df,
            "tip": self.tips_df,
            "discussion": self.discussions_df,
            "community": self.communities_df,
        }

        id_col = ID_COLUMN_MAP[content_type]
        content_df = df_map[content_type]

        top_ids = (
            self.interactions_df[
                (self.interactions_df["type"] == "like")
                & (self.interactions_df["item_type"] == content_type)
            ]
            .groupby("item_id")
            .size()
            .sort_values(ascending=False)
            .head(n)
            .index
        )

        return content_df[content_df[id_col].isin(top_ids)]
