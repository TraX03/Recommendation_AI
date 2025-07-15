from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.filtering_utils import filter_avoid_ingredients


class ColdStartService:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words="english")

    def generate_coldstart(
        self,
        user_prefs: dict,
        df: pd.DataFrame,
        id_column: str,
        text_column: str = "combined_text",
        avoid_filter: bool = False,
        fit_new: bool = False,
        max_recs: int = 10,
    ) -> pd.DataFrame:
        text_data = df[text_column].fillna("")
        tfidf_matrix = (
            self.tfidf.fit_transform(text_data)
            if fit_new
            else self.tfidf.transform(text_data)
        )

        profile_keywords = user_prefs.get("diet", []) + user_prefs.get(
            "region_pref", []
        )
        user_vector = self.tfidf.transform([" ".join(profile_keywords)])
        similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

        ranked_df = df.iloc[similarities.argsort()[::-1]].copy()
        ranked_df["similarity"] = similarities[similarities.argsort()[::-1]]

        if avoid_filter:
            ranked_df = self._recipe_avoid_filter(ranked_df, user_prefs)

        preferred_mask = ranked_df[text_column].apply(
            lambda text: self._is_preferred_text(
                text,
                user_prefs.get("diet", []),
                user_prefs.get("region_pref", []),
            )
        )

        preferred = ranked_df[preferred_mask]
        fallback = ranked_df[~preferred_mask]

        return (
            pd.concat([preferred.head(max_recs), fallback.head(max_recs)])
            .drop_duplicates(subset=[id_column])
            .head(max_recs)
        )

    def _is_preferred_text(self, text: str, diet: List[str], region: List[str]) -> bool:
        text = text.lower()
        matches_diet = any(d.lower() in text for d in diet) if diet else True
        matches_region = any(r.lower() in text for r in region) if region else True
        return matches_diet and matches_region

    def _recipe_avoid_filter(self, df: pd.DataFrame, prefs: dict) -> pd.DataFrame:
        df = filter_avoid_ingredients(df, prefs.get("avoid_ingredients", []))
        return df[
            ~df["category"].apply(
                lambda cat: "sauce" in " ".join(cat).lower()
                if isinstance(cat, list)
                else "sauce" in str(cat).lower()
            )
        ]
