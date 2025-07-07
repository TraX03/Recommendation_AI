from typing import List

import pandas as pd

from app.constants import BLOCKED_KEYWORDS_BY_DIET, RECOMMENDATION_DATA_COLLECTION_ID
from app.utils.appwrite_client import get_document_by_id


def filter_recent_seen(df: pd.DataFrame, user_id: str, id_col: str) -> pd.DataFrame:
    try:
        doc = get_document_by_id(
            collection_id=RECOMMENDATION_DATA_COLLECTION_ID,
            document_id=user_id,
        )
        recent_ids = {str(rid) for rid in doc.get("last_recommendations", [])[:60]}
    except Exception:
        recent_ids = set()

    df[id_col] = df[id_col].astype(str)

    return df[~df[id_col].isin(recent_ids)]


def filter_avoid_ingredients(
    df: pd.DataFrame, avoid_ingredients: List[str]
) -> pd.DataFrame:
    if not avoid_ingredients:
        return df

    avoid_lower = [ingredient.lower() for ingredient in avoid_ingredients]

    def should_filter(text: str) -> bool:
        text_lower = text.lower()
        return any(ingredient in text_lower for ingredient in avoid_lower)

    return df[~df["combined_text"].apply(should_filter)]


def filter_diet(df: pd.DataFrame, diet_input) -> pd.DataFrame:
    if not diet_input or "combined_text" not in df.columns:
        return df

    if isinstance(diet_input, str):
        diet_input = [diet_input]

    diet_input = [d.strip().lower() for d in diet_input]
    blocked_terms = set(
        term for diet in diet_input for term in BLOCKED_KEYWORDS_BY_DIET.get(diet, [])
    )

    def should_include(text: str) -> bool:
        text_lower = str(text).lower()
        return any(diet in text_lower for diet in diet_input) or not any(
            term in text_lower for term in blocked_terms
        )

    return df[df["combined_text"].apply(should_include)]
