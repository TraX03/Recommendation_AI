from typing import List

import pandas as pd

from app.utils.shelve_utils import get_last_recommendations


def filter_recent_seen(df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    """Filter out recipes recently seen by the user."""
    seen = {
        rid
        for round_recipes in get_last_recommendations(user_id)
        for rid in round_recipes[:60]
    }
    return df[~df["recipe_id"].isin(seen)]


def filter_avoid_ingredients(
    df: pd.DataFrame, avoid_ingredients: List[str]
) -> pd.DataFrame:
    """Filter out recipes that contain ingredients the user wants to avoid."""
    if not avoid_ingredients:
        return df

    avoid_lower = [ingredient.lower() for ingredient in avoid_ingredients]

    def should_filter(text: str) -> bool:
        text_lower = text.lower()
        return any(ingredient in text_lower for ingredient in avoid_lower)

    return df[~df["combined_text"].apply(should_filter)]
