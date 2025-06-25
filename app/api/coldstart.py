from fastapi import APIRouter, HTTPException

from app import dependencies
from app.models.response_models import RecommendationResponse
from app.services.coldstart_service import generate_coldstart_recommendations
from app.utils.appwrite_client import get_user_document

coldstart_router = APIRouter()


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


@coldstart_router.post("/coldstart/{user_id}", response_model=RecommendationResponse)
def cold_start_from_user(user_id: str):
    prefs = get_user_preferences(user_id)

    recommended_df = generate_coldstart_recommendations(
        user_id=user_id,
        user_prefs=prefs,
        recipes_df=dependencies.recipes_df,
        tfidf_matrix=dependencies.tfidf_matrix,
        tfidf_vectorizer=dependencies.tfidf_vectorizer,
        max_recs=10,
    )

    return RecommendationResponse(
        recipe_ids=recommended_df["recipe_id"].tolist(),
        titles=recommended_df["title"].tolist(),
        images=recommended_df["image"].tolist(),
    )
