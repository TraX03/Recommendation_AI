from typing import List

import pandas as pd

from app.constants import BLOCKED_KEYWORDS_BY_DIET, RECOMMENDATION_DATA_COLLECTION_ID
from app.utils.appwrite_client import get_document_by_id


def filter_recent_seen(
    df: pd.DataFrame,
    user_id: str,
    id_col: str,
    content_type: str,
    max_recent: int = 50,
) -> pd.DataFrame:
    key = f"last_recommendations_{content_type}"
    try:
        doc = get_document_by_id(RECOMMENDATION_DATA_COLLECTION_ID, user_id)
        recent_ids = set(str(rid) for rid in doc.get(key, [])[:max_recent])
    except Exception:
        recent_ids = set()

    seen_mask = df[id_col].astype(str).isin(recent_ids)
    return df[~seen_mask]


def filter_avoid_ingredients(
    df: pd.DataFrame, avoid_ingredients: List[str]
) -> pd.DataFrame:
    if not avoid_ingredients:
        return df

    avoid_lower = [ingredient.lower() for ingredient in avoid_ingredients]

    return df[
        ~df["combined_text"]
        .str.lower()
        .apply(lambda text: any(ingredient in text for ingredient in avoid_lower))
    ]


def filter_diet(df: pd.DataFrame, diet_input: List[str]) -> pd.DataFrame:
    if not diet_input:
        return df

    diet_input = [
        d.strip().lower()
        for d in (diet_input if isinstance(diet_input, list) else [diet_input])
    ]
    blocked_terms = set(
        term for diet in diet_input for term in BLOCKED_KEYWORDS_BY_DIET.get(diet, [])
    )

    def is_valid(text: str) -> bool:
        text_lower = text.lower()
        return any(diet in text_lower for diet in diet_input) or not any(
            term in text_lower for term in blocked_terms
        )

    return df[df["combined_text"].str.lower().apply(is_valid)]


def filter_region(df: pd.DataFrame, region_pref: List[str]) -> pd.DataFrame:
    if not region_pref:
        return df

    region_keywords = [region.lower() for region in region_pref]

    return df[
        df["combined_text"]
        .str.lower()
        .apply(lambda text: any(region in text for region in region_keywords))
    ]
