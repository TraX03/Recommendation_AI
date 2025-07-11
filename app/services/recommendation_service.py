import json
import random
import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
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
from app.utils.filtering_utils import (
    filter_avoid_ingredients,
    filter_diet,
    filter_recent_seen,
)

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_local_embeddings(texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
    return list(model.encode(texts, batch_size=batch_size, convert_to_numpy=True))


def build_tfidf_and_similarity(df: pd.DataFrame, id_col: str):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_text"].fillna(""))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df[id_col]).drop_duplicates()

    return {
        "tfidf_matrix": tfidf_matrix,
        "cosine_sim": cosine_sim,
        "indices": indices,
        "vectorizer": tfidf,
    }


class HybridRecommender:
    def __init__(
        self,
        recipes_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        tips_df: pd.DataFrame = pd.DataFrame(),
        discussions_df: pd.DataFrame = pd.DataFrame(),
        communities_df: pd.DataFrame = pd.DataFrame(),
    ):
        self.data_map = {
            "recipe": recipes_df,
            "tip": tips_df,
            "discussion": discussions_df,
            "community": communities_df,
        }
        self.interactions_df = interactions_df
        self.similarity_models = {}
        self.sim_matrices = {}
        self._prepare_content_based_models()
        self._prepare_collaborative_models()

    def _prepare_content_based_models(self):
        for ctype, config in CONTENT_TYPE_MAP.items():
            df = self.data_map.get(ctype)
            if df.empty:
                continue

            tfidf_model = build_tfidf_and_similarity(df, config["id_col"])

            if "embedding" in df.columns and df["embedding"].notna().any():
                embedding_sim = cosine_similarity(np.stack(df["embedding"].dropna()))
                indices = pd.Series(
                    df.index, index=df[config["id_col"]]
                ).drop_duplicates()
                self.similarity_models[ctype] = {
                    "cosine_sim": embedding_sim,
                    "indices": indices,
                    "vectorizer": tfidf_model["vectorizer"],
                    "tfidf_matrix": tfidf_model["tfidf_matrix"],
                }
            else:
                self.similarity_models[ctype] = tfidf_model

    def _prepare_collaborative_models(self):
        self.interactions_df["score"] = self.interactions_df.apply(
            self._normalize_score, axis=1
        )
        self.interactions_df = pd.concat(
            [
                self.interactions_df[
                    ~self.interactions_df["type"].isin(["view", "follow"])
                ],
                self._generate_view_scores(),
            ]
        )

        for ctype in self.data_map:
            self.sim_matrices[ctype] = self._build_cf_matrix(ctype)

    def _normalize_score(self, row) -> float:
        score = row.get("score")
        if row["type"] == "rating" and pd.notnull(score):
            return float(score) / 10.0
        return (
            float(score)
            if pd.notnull(score)
            else INTERACTION_WEIGHTS.get(row["type"], 0.0)
        )

    def _generate_view_scores(self) -> pd.DataFrame:
        recent_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        views = self.interactions_df[self.interactions_df["type"] == "view"].explode(
            "timestamps"
        )
        views["timestamp"] = pd.to_datetime(views["timestamps"], utc=True)
        recent_views = views[views["timestamp"] >= recent_cutoff]

        counts = (
            recent_views.groupby(["user_id", "item_id", "item_type"])
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

    def _build_cf_matrix(self, ctype: str):
        df = self.interactions_df[self.interactions_df["item_type"] == ctype]
        if df.empty:
            return {}

        user_map = {uid: i for i, uid in enumerate(df["user_id"].unique())}
        item_map = {iid: i for i, iid in enumerate(df["item_id"].unique())}
        if not user_map or not item_map:
            return {}

        df = df.copy()
        df["user_idx"] = df["user_id"].map(user_map)
        df["item_idx"] = df["item_id"].map(item_map)

        matrix = csr_matrix((df["score"], (df["user_idx"], df["item_idx"])))
        return {
            "matrix": cosine_similarity(matrix.T),
            "item_map": item_map,
            "item_idx_to_id": {v: k for k, v in item_map.items()},
        }

    def _build_user_profile_vector(
        self, user_id: str, content_model, prefs: dict
    ) -> np.ndarray:
        user_interactions = self.interactions_df.query("user_id == @user_id")

        strong = user_interactions[
            (user_interactions["type"].isin(["bookmark", "like"]))
            | (user_interactions.get("rating", 0) >= 7.0)
        ]

        keywords = []
        if not strong.empty:
            item_ids = strong["item_id"].unique()
            id_col = CONTENT_TYPE_MAP["recipe"]["id_col"]
            matched = self.data_map["recipe"][
                self.data_map["recipe"][id_col].isin(item_ids)
            ]

            for _, row in matched.iterrows():
                keywords += (
                    row.get("tags", [])
                    + row.get("title", "").lower().split()
                    + row.get("description", "").lower().split()
                )

        for key in ["diet", "region_pref"]:
            values = prefs.get(key)
            if isinstance(values, list):
                keywords += values
            elif isinstance(values, str):
                keywords.append(values)

        clean_keywords = [
            re.sub(r"[\s\-]", "", str(k).strip().lower())
            for k in keywords
            if isinstance(k, str) and k.strip()
        ]

        if not clean_keywords or content_model.get("vectorizer") is None:
            return np.zeros(content_model["tfidf_matrix"].shape[0])

        profile_text = " ".join(clean_keywords)
        user_vector = content_model["vectorizer"].transform([profile_text])
        return cosine_similarity(user_vector, content_model["tfidf_matrix"]).flatten()

    def get_hybrid_recommendations(
        self,
        user_id: str,
        item_id: str,
        content_type: str,
        user_prefs: Optional[dict] = None,
        item_cbf_weight=0.4,
        user_cbf_weight=0.3,
        cf_weight=0.3,
        top_k=100,
        sample_n=40,
    ) -> pd.DataFrame:
        model = self.similarity_models.get(content_type)
        if not model or item_id not in model["indices"]:
            return pd.DataFrame()

        # Item profile CBF scoring
        cbf_idx = model["indices"][item_id]
        item_scores = minmax_scale(model["cosine_sim"][cbf_idx])

        # CF scoring
        cf_matrix_data = self.sim_matrices.get(content_type)
        cf_scores = np.zeros_like(item_scores)
        if cf_matrix_data and item_id in cf_matrix_data["item_map"]:
            cf_idx = cf_matrix_data["item_map"][item_id]
            cf_partial = cf_matrix_data["matrix"][cf_idx]
            for rid, partial_idx in cf_matrix_data["item_map"].items():
                full_idx = model["indices"].get(rid)
                if full_idx is not None:
                    cf_scores[full_idx] = cf_partial[partial_idx]
            cf_scores = minmax_scale(cf_scores)

        # User profile CBF scoring
        user_scores = minmax_scale(
            self._build_user_profile_vector(user_id, model, user_prefs or {})
        )
        user_scores = minmax_scale(user_scores)

        hybrid_scores = (
            item_cbf_weight * item_scores
            + user_cbf_weight * user_scores
            + cf_weight * cf_scores
        )

        top_indices = np.argsort(-hybrid_scores)[:top_k]
        selected = random.sample(list(top_indices), min(sample_n, len(top_indices)))

        df = self.data_map[content_type]
        if content_type == "community" and "name" in df and "title" not in df:
            df["title"] = df["name"]

        for col in ["title", "image", "author_id"]:
            if col not in df.columns:
                df[col] = ""

        return df.iloc[selected][
            [
                CONTENT_TYPE_MAP[content_type]["id_col"],
                "title",
                "image",
                "author_id",
                "combined_text",
            ]
        ]

    def get_trending_items(self, content_type: str, n: int = 20) -> pd.DataFrame:
        df = self.data_map[content_type]
        id_col = CONTENT_TYPE_MAP[content_type]["id_col"]
        top_ids = (
            self.interactions_df.query("type == 'like' and item_type == @content_type")
            .groupby("item_id")
            .size()
            .nlargest(n)
            .index
        )
        return df[df[id_col].isin(top_ids)]

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
        top_up = [i for i in ids if i not in seeds][: threshold - len(seeds)]
        return seeds + top_up

    def generate_recommendations(
        self,
        user_id: str,
        content_type: str,
        prefs: dict,
        interactions: pd.DataFrame,
        max_count: int,
    ) -> PostList:
        id_col = CONTENT_TYPE_MAP[content_type]["id_col"]

        trending_ids = self.get_trending_items(content_type, n=10)[id_col].tolist()
        recent_ids = (
            interactions.query("item_type == @content_type")["item_id"]
            .dropna()
            .unique()
            .tolist()[:10]
        )

        seeds = recent_ids + [tid for tid in trending_ids if tid not in recent_ids][:2]
        threshold = 5 if content_type == "community" else 10
        if len(seeds) < threshold:
            seeds = self._coldstart_seeds(user_id, content_type, seeds)

        recs = pd.concat(
            [
                self.get_hybrid_recommendations(
                    user_id=user_id,
                    item_id=sid,
                    content_type=content_type,
                    user_prefs=prefs,
                    item_cbf_weight=0.3,
                    user_cbf_weight=0.4,
                    cf_weight=0.3,
                    top_k=100,
                    sample_n=max_count,
                )
                for sid in seeds
            ]
        ).drop_duplicates(subset=id_col)

        recs = filter_recent_seen(recs, user_id, id_col)
        if content_type == "recipe":
            recs = filter_avoid_ingredients(recs, prefs.get("avoid_ingredients", []))
            recs = filter_diet(recs, prefs.get("diet"))

        final = recs.head(max_count)
        create_or_update_document(
            RECOMMENDATION_DATA_COLLECTION_ID,
            user_id,
            {"user_id": user_id, "last_recommendations": final[id_col].tolist()},
        )

        return PostList(
            post_ids=final[id_col].tolist(),
            titles=final["title"].tolist(),
            images=final["image"].tolist(),
            author_ids=final["author_id"].tolist(),
        )
