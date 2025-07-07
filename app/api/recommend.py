import pandas as pd
from fastapi import APIRouter

from app import dependencies
from app.models.response_models import RecommendationResponse
from app.services.recommendation_service import HybridRecommender
from app.utils.load_utils import get_user_preferences

recommend_router = APIRouter()


@recommend_router.post("/homeFeed/{user_id}", response_model=RecommendationResponse)
def recommend_for_home(user_id: str) -> RecommendationResponse:
    engine: HybridRecommender = dependencies.engine
    prefs = get_user_preferences(user_id)

    interactions = engine.interactions_df.query("user_id == @user_id").copy()
    interactions["created_at"] = pd.to_datetime(
        interactions["created_at"], errors="coerce"
    )
    interactions = interactions.sort_values(by="created_at", ascending=False)

    return RecommendationResponse(
        recipe=engine.generate_recommendations(
            user_id, "recipe", prefs, interactions, 100
        ),
        tip=engine.generate_recommendations(user_id, "tip", prefs, interactions, 100),
        discussion=engine.generate_recommendations(
            user_id, "discussion", prefs, interactions, 100
        ),
        community=engine.generate_recommendations(
            user_id, "community", prefs, interactions, 30
        ),
    )
