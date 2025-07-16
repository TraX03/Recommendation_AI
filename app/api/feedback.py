from fastapi import APIRouter

from app import dependencies
from app.models.schemas import FeedbackRequest
from app.utils.data_loader import fetch_session_data, get_user_preferences

feedback_router = APIRouter()


@feedback_router.post("/homeFeed/{user_id}")
def log_home_session_feedback(user_id: str) -> dict:
    engine = dependencies.engine
    engine.refresh_models()
    prefs = get_user_preferences(user_id)

    result = engine.log_session_feedback(user_id, prefs)
    return {"detail": result}


@feedback_router.post("/mealplan/{user_id}")
def log_inventory_feedback(
    user_id: str,
    payload: FeedbackRequest,
) -> dict:
    engine = dependencies.engine
    engine.refresh_models()

    session_data = fetch_session_data(
        user_id=user_id,
        date=payload.date,
        mealtime=payload.mealtime,
        target_recipe_id=payload.target_recipe_id,
    )

    result = engine.log_inventory_feedback(
        session_data=session_data,
        is_regenerate=payload.is_regenerate,
    )

    return {"detail": result}
