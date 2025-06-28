import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.filtering_utils import filter_avoid_ingredients


def generate_coldstart_recommendations(
    user_prefs: dict,
    recipes_df: pd.DataFrame,
    tfidf_matrix,
    tfidf_vectorizer,
    max_recs: int = 10,
) -> pd.DataFrame:
    profile_text = " ".join(
        user_prefs.get("diet", []) + user_prefs.get("region_pref", [])
    )
    user_vector = tfidf_vectorizer.transform([profile_text])

    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    ranked_indices = similarities.argsort()[::-1]
    ranked_df = recipes_df.iloc[ranked_indices].copy()
    ranked_df["similarity"] = similarities[ranked_indices]

    filtered_df = filter_avoid_ingredients(
        ranked_df, user_prefs.get("avoid_ingredients", [])
    )

    def is_preferred(text: str) -> bool:
        text = text.lower()
        matches_diet = (
            any(d.lower() in text for d in user_prefs.get("diet", []))
            if user_prefs.get("diet")
            else True
        )
        matches_region = (
            any(r.lower() in text for r in user_prefs.get("region_pref", []))
            if user_prefs.get("region_pref")
            else True
        )
        return matches_diet and matches_region

    preferred = filtered_df[filtered_df["combined_text"].apply(is_preferred)]
    fallback = filtered_df[~filtered_df["combined_text"].apply(is_preferred)]

    final_df = (
        pd.concat([preferred.head(max_recs), fallback.head(max_recs)])
        .drop_duplicates()
        .head(max_recs)
    )

    return final_df
