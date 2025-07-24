import re
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.constants import CONTENT_TYPE_MAP


class ContentBasedService:
    def prepare_cbf_models(self, data_map: dict) -> dict:
        return {
            ctype: self._build_similarity_model(df, CONTENT_TYPE_MAP[ctype]["id_col"])
            for ctype, df in data_map.items()
        }

    def _build_similarity_model(self, df: pd.DataFrame, id_col: str) -> Dict:
        tfidf_model = self.build_tfidf_model(df, id_col)
        embedding_model = self._build_embedding_similarity(df, id_col)

        if embedding_model:
            return {
                "cosine_sim": embedding_model["cosine_sim"],
                "indices": embedding_model["indices"],
                "vectorizer": tfidf_model["vectorizer"],
                "tfidf_matrix": tfidf_model["tfidf_matrix"],
            }
        else:
            return tfidf_model

    def build_tfidf_model(self, df: pd.DataFrame, id_col: str) -> Dict:
        texts = df["combined_text"].fillna("").astype(str).tolist()
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(texts)
        cosine_sim = cosine_similarity(tfidf_matrix)
        indices = pd.Series(df.index, index=df[id_col]).drop_duplicates()

        return {
            "tfidf_matrix": tfidf_matrix,
            "cosine_sim": cosine_sim,
            "indices": indices,
            "vectorizer": tfidf,
        }

    def _build_embedding_similarity(
        self, df: pd.DataFrame, id_col: str
    ) -> Optional[Dict]:
        if "embedding" not in df.columns or df["embedding"].isna().all():
            return None

        embeddings = np.stack(df["embedding"].dropna())
        cosine_sim = cosine_similarity(embeddings)
        indices = pd.Series(df.index, index=df[id_col]).drop_duplicates()

        return {"cosine_sim": cosine_sim, "indices": indices}

    def build_user_profile_vector(
        self,
        user_interactions: pd.DataFrame,
        data_map: Dict[str, pd.DataFrame],
        prefs: Dict,
        tfidf_model: Dict,
    ) -> np.ndarray:
        if "score" in user_interactions.columns:
            strong = user_interactions[
                (user_interactions["type"].isin(["bookmark", "like"]))
                | (user_interactions["score"].fillna(0) >= 7.0)
            ]
        else:
            strong = user_interactions[
                user_interactions["type"].isin(["bookmark", "like"])
            ]

        keywords = []

        for content_type in ["recipe", "tip", "discussion"]:
            if content_type not in data_map:
                continue

            if not strong.empty:
                item_ids = strong["item_id"].unique()
                id_col = CONTENT_TYPE_MAP.get(content_type, {}).get("id_col")
                if not id_col:
                    continue
                matched = data_map[content_type][
                    data_map[content_type][id_col].isin(item_ids)
                ]

                for _, row in matched.iterrows():
                    tags = row.get("tags", [])

                    if not isinstance(tags, list):
                        tags = [tags]

                    keywords += tags

                    title = row.get("title", "").lower().split()
                    description = row.get("description", "").lower().split()

                    keywords += title + description

        for key in ["diet", "region_pref", "tags"]:
            values = prefs.get(key)
            if isinstance(values, list):
                keywords += values
            elif isinstance(values, str):
                keywords.append(values)

        clean_keywords = [
            re.sub(r"\s+", " ", str(k).strip().lower())
            for k in keywords
            if isinstance(k, str) and k.strip()
        ]

        if not clean_keywords or tfidf_model.get("vectorizer") is None:
            return np.zeros(tfidf_model["tfidf_matrix"].shape[0])

        profile_text = " ".join(clean_keywords)
        user_vector = tfidf_model["vectorizer"].transform([profile_text])
        similarity = cosine_similarity(
            user_vector, tfidf_model["tfidf_matrix"]
        ).flatten()

        return similarity
