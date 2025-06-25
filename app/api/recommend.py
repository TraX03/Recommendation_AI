import pandas as pd
from fastapi import APIRouter, HTTPException

from app.models.response_models import RecommendationResponse
from app.services.recommendation_service import RecommendationEngine
from app.utils.appwrite_client import get_user_document
from app.utils.filtering_utils import filter_avoid_ingredients, filter_recent_seen
from app.utils.shelve_utils import save_last_recommendations

recommend_router = APIRouter()


def get_user_preferences(user_id: str) -> dict:
    try:
        user = get_user_document(user_id)
        return {
            "avoid_ingredients": user.get("avoid_ingredients", []),
            "diet": user.get("diet", []),
            "region_pref": user.get("region_pref", []),
        }
    except Exception:
        raise HTTPException(status_code=404, detail="User not found")


@recommend_router.post("/homeFeed/{user_id}", response_model=RecommendationResponse)
def recommend_for_home(user_id: str):
    global engine
    prefs = get_user_preferences(user_id)

    # Step 1: Choose seed recipes
    user_interactions = engine.interactions_df[
        engine.interactions_df["user_id"] == user_id
    ]
    user_interactions = user_interactions.sort_values(by="created_at", ascending=False)
    recent_recipes = user_interactions["recipe_id"].dropna().unique().tolist()[:10]
    trending_recipes = engine.get_trending_recipes(n=10)["recipe_id"].tolist()
    trending_filtered = [r for r in trending_recipes if r not in recent_recipes][:2]
    seed_recipes = recent_recipes + trending_filtered

    if not seed_recipes:
        seed_recipes = engine.get_trending_recipes(n=10)["recipe_id"].tolist()

    # Step 2: Get new recommendations
    results = []
    for rid in seed_recipes:
        cbf_weight, cf_weight = engine.adaptive_weights(rid)
        recs = engine.get_hybrid_recommendations(
            rid, cbf_weight, cf_weight, top_k=50, sample_n=20
        )
        results.append(recs)

    combined_df = pd.concat(results).drop_duplicates(subset="recipe_id")

    # Step 3: Filter seen recipes and avoid ingredients
    fresh_df = filter_recent_seen(combined_df, user_id)
    filtered_df = filter_avoid_ingredients(fresh_df, prefs["avoid_ingredients"])

    # Step 4: Save and return
    save_last_recommendations(user_id, filtered_df["recipe_id"].tolist())

    return RecommendationResponse(
        recipe_ids=filtered_df["recipe_id"].tolist(),
        titles=filtered_df["title"].tolist(),
        images=filtered_df["image"].tolist(),
        author_ids=filtered_df["author_id"].tolist(),
    )
