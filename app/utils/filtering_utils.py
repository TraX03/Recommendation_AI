from typing import List

import pandas as pd

from app.constants import RECOMMENDATION_DATA_COLLECTION_ID
from app.utils.appwrite_client import get_document_by_id


def filter_recent_seen(df: pd.DataFrame, user_id: str, id_col: str) -> pd.DataFrame:
    try:
        doc = get_document_by_id(
            collection_id=RECOMMENDATION_DATA_COLLECTION_ID,
            document_id=user_id,
        )
        recent_ids = set(doc.get("last_recommendations", []))
    except Exception:
        recent_ids = set()

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
