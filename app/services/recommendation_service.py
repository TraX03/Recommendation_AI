import json
import random
from datetime import datetime, timedelta, timezone
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale

from app.constants import (
    CONTENT_TYPE_MAP,
    INTERACTION_WEIGHTS,
    RECOMMENDATION_DATA_COLLECTION_ID,
)
from app.models.response_models import PostList
from app.utils.appwrite_client import create_or_update_document, get_document_by_id
from app.utils.filtering_utils import filter_avoid_ingredients, filter_recent_seen


def build_tfidf_and_similarity(df: pd.DataFrame, id_col: str):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_text"].fillna(""))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df[id_col]).drop_duplicates()
    return {"tfidf_matrix": tfidf_matrix, "cosine_sim": cosine_sim, "indices": indices}


class RecommendationEngine:
    def __init__(
        self,
        recipes_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        tips_df: pd.DataFrame = pd.DataFrame(),
        discussions_df: pd.DataFrame = pd.DataFrame(),
        communities_df: pd.DataFrame = pd.DataFrame(),
        data_sufficient: bool = False,
    ):
        self.recipes_df = recipes_df
        self.interactions_df = interactions_df
        self.tips_df = tips_df
        self.discussions_df = discussions_df
        self.communities_df = communities_df
        self.data_sufficient = data_sufficient

        self.similarity_models = {k: {} for k in CONTENT_TYPE_MAP}
        self.sim_matrices = {k: {} for k in CONTENT_TYPE_MAP}

        self._preprocess_content_based()
        self._preprocess_collaborative_based()

    def _preprocess_content_based(self):
        for ctype, config in CONTENT_TYPE_MAP.items():
            df = getattr(self, config["attr"])
            if not df.empty:
                self.similarity_models[ctype] = build_tfidf_and_similarity(
                    df, config["id_col"]
                )

    def _preprocess_collaborative_based(self):
        now = datetime.now(timezone.utc)
        recent_cutoff = now - timedelta(days=7)

        self.interactions_df["score"] = self.interactions_df.apply(
            self._normalize_score,
            axis=1,
        )

        view_scores = self._compute_view_decay_scores(recent_cutoff)
        explicit = self.interactions_df[
            ~self.interactions_df["type"].isin(["view", "follow"])
        ]
        self.interactions_df = pd.concat([explicit, view_scores], ignore_index=True)

        for ctype in ["recipe", "discussion", "tip", "community"]:
            self.sim_matrices[ctype] = self._build_cf_similarity_matrix(ctype)

    def _normalize_score(self, row) -> float:
        if row["type"] == "rating" and pd.notnull(row.get("score")):
            return float(row["score"]) / 10.0
        elif pd.notnull(row.get("score")):
            return float(row["score"])
        else:
            return INTERACTION_WEIGHTS.get(row["type"], 0.0)

    def _compute_view_decay_scores(self, since: datetime) -> pd.DataFrame:
        view_df = self.interactions_df[self.interactions_df["type"] == "view"].explode(
            "timestamps"
        )
        view_df["timestamp"] = pd.to_datetime(view_df["timestamps"], utc=True)
        recent = view_df[view_df["timestamp"] >= since]

        counts = (
            recent.groupby(["user_id", "item_id", "item_type"])
            .agg(view_count=("timestamp", "count"), created_at=("timestamp", "max"))
            .reset_index()
        )
        counts["score"] = counts["view_count"].apply(
            lambda c: min(0.2 * np.log1p(c), 0.6)
        )
        counts["type"] = "view"
        return counts[
            ["user_id", "item_id", "item_type", "type", "score", "created_at"]
        ]

    def _build_cf_similarity_matrix(self, content_type: str):
        df = self.interactions_df[self.interactions_df["item_type"] == content_type]
        if df.empty:
            return {}

        df = df.copy()

        user_map = {uid: i for i, uid in enumerate(df["user_id"].unique())}
        item_map = {iid: i for i, iid in enumerate(df["item_id"].unique())}
        if not user_map or not item_map:
            return {}

        df["user_idx"] = df["user_id"].map(user_map)
        df["item_idx"] = df["item_id"].map(item_map)

        interaction_matrix = csr_matrix(
            (df["score"], (df["user_idx"], df["item_idx"])),
            shape=(len(user_map), len(item_map)),
        )

        return {
            "matrix": cosine_similarity(interaction_matrix.T),
            "item_map": item_map,
            "item_idx_to_id": {v: k for k, v in item_map.items()},
        }

    def adaptive_weights(self, item_id: str) -> Tuple[float, float]:
        count = self.interactions_df[
            self.interactions_df["item_id"] == str(item_id)
        ].shape[0]
        total = len(self.interactions_df)
        cf_weight = min(0.8, count / total + 0.1 if total else 0)
        return 1.0 - cf_weight, cf_weight

    def get_hybrid_recommendations(
        self,
        item_id: str,
        content_type: str,
        cbf_weight=0.6,
        cf_weight=0.4,
        top_k=100,
        sample_n=40,
    ) -> pd.DataFrame:
        model = self.similarity_models.get(content_type)
        if not model or item_id not in model["indices"]:
            return pd.DataFrame()

        cbf_idx = model["indices"][item_id]
        cbf_scores = minmax_scale(model["cosine_sim"][cbf_idx])

        cf_data = self.sim_matrices.get(content_type)
        if cf_data and item_id in cf_data["item_map"]:
            cf_idx = cf_data["item_map"][item_id]
            cf_partial = cf_data["matrix"][cf_idx]

            cf_scores = np.zeros_like(cbf_scores)
            for rid, partial_idx in cf_data["item_map"].items():
                full_idx = model["indices"].get(rid)
                if full_idx is not None:
                    cf_scores[full_idx] = cf_partial[partial_idx]

            cf_scores = minmax_scale(cf_scores)
            hybrid = cbf_weight * cbf_scores + cf_weight * cf_scores
        else:
            hybrid = cbf_scores

        sim_scores = sorted(
            [(i, s) for i, s in enumerate(hybrid) if i != cbf_idx],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        selected = random.sample(sim_scores, min(sample_n, len(sim_scores)))
        indices = [i for i, _ in selected]

        config = CONTENT_TYPE_MAP[content_type]
        df = getattr(self, config["attr"]).copy()
        id_col = config["id_col"]

        if (
            content_type == "community"
            and "name" in df.columns
            and "title" not in df.columns
        ):
            df["title"] = df["name"]

        for col in ["title", "image", "author_id"]:
            if col not in df.columns:
                df[col] = ""

        return df.iloc[indices][[id_col, "title", "image", "author_id"]]

    def get_trending_items(self, content_type: str, n: int = 20) -> pd.DataFrame:
        config = CONTENT_TYPE_MAP[content_type]
        id_col = config["id_col"]
        content_df = getattr(self, config["attr"])

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

    def generate_recommendations(
        self,
        user_id: str,
        content_type: str,
        prefs: dict,
        interactions: pd.DataFrame,
        max_count: int,
    ) -> PostList:
        config = CONTENT_TYPE_MAP[content_type]
        id_col = config["id_col"]

        trending_ids = self.get_trending_items(content_type, n=10)[id_col].tolist()
        recent_ids = (
            interactions[interactions["item_type"] == content_type]["item_id"]
            .dropna()
            .unique()
            .tolist()[:10]
        )
        seeds = recent_ids + [t for t in trending_ids if t not in recent_ids][:2]

        seed_threshold = 5 if content_type == "community" else 10
        if len(seeds) < seed_threshold:
            seeds = self._coldstart_seeds(user_id, content_type, seeds)

        all_recs = [
            self.get_hybrid_recommendations(
                sid,
                content_type,
                *self.adaptive_weights(sid) if self.data_sufficient else (0.6, 0.4),
                top_k=100,
                sample_n=max_count,
            )
            for sid in seeds
        ]

        combined = pd.concat(all_recs).drop_duplicates(subset=id_col)
        filtered = filter_recent_seen(combined, user_id, id_col)

        if content_type == "recipe":
            filtered = filter_avoid_ingredients(
                filtered, prefs.get("avoid_ingredients", [])
            )

        final = filtered.head(max_count)
        create_or_update_document(
            collection_id=RECOMMENDATION_DATA_COLLECTION_ID,
            document_id=user_id,
            data={"user_id": user_id, "last_recommendations": final[id_col].tolist()},
        )

        return PostList(
            post_ids=final[id_col].tolist(),
            titles=final["title"].tolist(),
            images=final["image"].tolist(),
            author_ids=final["author_id"].tolist(),
        )

    def _coldstart_seeds(self, user_id: str, content_type: str, seeds: list) -> list:
        key = {"tip": "tip", "discussion": "discussion", "community": "community"}.get(
            content_type
        )
        try:
            doc = get_document_by_id(RECOMMENDATION_DATA_COLLECTION_ID, user_id)
            onboarding = json.loads(doc.get("onboarding_suggestions", "{}"))
        except Exception:
            onboarding = {}

        ids = onboarding.get(key, []) if key else []
        threshold = 5 if content_type == "community" else 10

        if not seeds:
            return ids[:threshold]
        top_up = [i for i in ids if i not in seeds][: threshold - len(seeds)]
        return seeds + top_up
