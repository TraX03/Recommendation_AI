from datetime import datetime, timedelta, timezone
from typing import Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from app.constants import INTERACTION_WEIGHTS


class CollaborativeService:
    def prepare_cf_models(
        self, interactions_df: pd.DataFrame, content_types: list[str]
    ) -> Dict[str, Dict]:
        processed_df = self._prepare_interactions(interactions_df)

        sim_matrices = {
            ctype: self._build_cf_matrix(processed_df, ctype) for ctype in content_types
        }

        return sim_matrices

    def _prepare_interactions(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        if interactions_df.empty or not {"user_id", "item_id", "type"}.issubset(
            interactions_df.columns
        ):
            print(
                "[WARN] interactions_df is empty or missing required columns. Skipping collaborative prep."
            )
            return pd.DataFrame(
                columns=["user_id", "item_id", "item_type", "type", "score"]
            )

        interactions_df = interactions_df.copy()
        interactions_df["score"] = interactions_df.apply(self._normalize_score, axis=1)

        return pd.concat(
            [
                interactions_df[~interactions_df["type"].isin(["view", "follow"])],
                self._generate_view_scores(interactions_df),
            ],
            ignore_index=True,
        )

    def _normalize_score(self, row) -> float:
        score = row.get("score")
        if row["type"] == "rating" and pd.notnull(score):
            return float(score) / 10.0
        return (
            float(score)
            if pd.notnull(score)
            else INTERACTION_WEIGHTS.get(row["type"], 0.0)
        )

    def _generate_view_scores(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        recent_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        views = interactions_df[interactions_df["type"] == "view"].explode("timestamps")
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

    def _build_cf_matrix(
        self, interactions_df: pd.DataFrame, content_type: str
    ) -> Dict:
        df = interactions_df[interactions_df["item_type"] == content_type]
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
