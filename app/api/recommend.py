import pandas as pd
from fastapi import APIRouter, HTTPException

from app.constants import USERS_COLLECTION_ID
from app.models.response_models import RecommendationResponse
from app.utils.appwrite_client import get_document_by_id
from app.utils.filtering_utils import filter_avoid_ingredients, filter_recent_seen
from app.utils.shelve_utils import save_last_recommendations

recommend_router = APIRouter()


def get_user_preferences(user_id: str) -> dict:
    try:
        user = get_document_by_id(USERS_COLLECTION_ID, user_id)
        return {
            "avoid_ingredients": user.get("avoid_ingredients", []),
            "diet": user.get("diet", []),
            "region_pref": user.get("region_pref", []),
        }
    except Exception:
        raise HTTPException(status_code=404, detail="User not found")


@recommend_router.post("/homeFeed/{user_id}", response_model=RecommendationResponse)
def recommend_for_home(user_id: str) -> RecommendationResponse:
    global engine

    prefs = get_user_preferences(user_id)
    interactions = engine.interactions_df.query("user_id == @user_id").copy()
    interactions["created_at"] = pd.to_datetime(
        interactions["created_at"], errors="coerce"
    )
    interactions = interactions.sort_values(by="created_at", ascending=False)

    recent = interactions["recipe_id"].dropna().unique().tolist()[:10]
    trending = engine.get_trending_recipes(n=10)["recipe_id"].tolist()
    trending_extras = [r for r in trending if r not in recent][:2]
    seeds = recent + trending_extras or trending

    all_recommendations = [
        engine.get_hybrid_recommendations(
            seed_id, *engine.adaptive_weights(seed_id), top_k=100, sample_n=40
        )
        for seed_id in seeds
    ]

    combined_df = pd.concat(all_recommendations).drop_duplicates(subset="recipe_id")

    filtered = filter_avoid_ingredients(
        filter_recent_seen(combined_df, user_id), prefs["avoid_ingredients"]
    ).head(100)

    save_last_recommendations(user_id, filtered["recipe_id"].tolist())

    return RecommendationResponse(
        recipe_ids=filtered["recipe_id"].tolist(),
        titles=filtered["title"].tolist(),
        images=filtered["image"].tolist(),
        author_ids=filtered["author_id"].tolist(),
    )
