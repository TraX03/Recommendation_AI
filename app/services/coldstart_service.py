from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.filtering_utils import filter_avoid_ingredients


def _is_preferred_text(text: str, diet: List[str], region: List[str]) -> bool:
    text = text.lower()
    matches_diet = any(d.lower() in text for d in diet) if diet else True
    matches_region = any(r.lower() in text for r in region) if region else True
    return matches_diet and matches_region


def generate_coldstart_df(
    user_prefs: dict,
    df: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer,
    tfidf_matrix,
    id_column: str,
    text_column: str,
    avoid_filter_fn=None,
    max_recs: int = 10,
) -> pd.DataFrame:
    profile_text = " ".join(
        user_prefs.get("diet", []) + user_prefs.get("region_pref", [])
    )
    user_vector = tfidf_vectorizer.transform([profile_text])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    ranked_indices = similarities.argsort()[::-1]
    ranked_df = df.iloc[ranked_indices].copy()
    ranked_df["similarity"] = similarities[ranked_indices]

    if avoid_filter_fn:
        ranked_df = avoid_filter_fn(ranked_df, user_prefs)

    preferred_mask = ranked_df[text_column].apply(
        lambda text: _is_preferred_text(
            text, user_prefs.get("diet", []), user_prefs.get("region_pref", [])
        )
    )

    preferred = ranked_df[preferred_mask]
    fallback = ranked_df[~preferred_mask]

    return (
        pd.concat([preferred.head(max_recs), fallback.head(max_recs)])
        .drop_duplicates(subset=[id_column])
        .head(max_recs)
    )


def generate_recipe_coldstart(
    user_prefs: dict,
    recipes_df: pd.DataFrame,
    tfidf_matrix,
    tfidf_vectorizer,
    max_recs: int = 10,
) -> pd.DataFrame:
    def recipe_avoid_filter(df: pd.DataFrame, prefs: dict) -> pd.DataFrame:
        df = filter_avoid_ingredients(df, prefs.get("avoid_ingredients", []))
        return df[
            ~df["category"].apply(
                lambda cat: "sauce" in " ".join(cat).lower()
                if isinstance(cat, list)
                else "sauce" in str(cat).lower()
            )
        ]

    return generate_coldstart_df(
        user_prefs=user_prefs,
        df=recipes_df,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        id_column="recipe_id",
        text_column="combined_text",
        avoid_filter_fn=recipe_avoid_filter,
        max_recs=max_recs,
    )


def generate_post_coldstart(
    user_prefs: dict,
    df: pd.DataFrame,
    tfidf_vectorizer,
    max_recs: int = 10,
    text_column: str = "combined_text",
    id_column: str = "post_id",
) -> pd.DataFrame:
    post_matrix = tfidf_vectorizer.transform(df[text_column].fillna(""))

    return generate_coldstart_df(
        user_prefs=user_prefs,
        df=df,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=post_matrix,
        id_column=id_column,
        text_column=text_column,
        max_recs=max_recs,
    )
