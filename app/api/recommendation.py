import json
from typing import List

from fastapi import APIRouter

from app import dependencies
from app.models.schemas import (
    MealItem,
    MealPlanRequest,
    MealPlanResponse,
    RecipeItem,
    RecommendationResponse,
)
from app.utils.data_loader import (
    fetch_session_data,
    get_mealplan_exclude_ids,
    get_user_preferences,
)
from app.utils.session_utils import start_session

recommend_router = APIRouter()


@recommend_router.post("/homeFeed/{user_id}", response_model=RecommendationResponse)
def recommend_for_home(user_id: str) -> RecommendationResponse:
    engine = dependencies.engine
    engine.refresh_models()
    prefs = get_user_preferences(user_id)
    start_session(user_id)

    return RecommendationResponse(
        recipe=engine.recommend(user_id, "recipe", prefs, 50),
        tip=engine.recommend(user_id, "tip", prefs, 50),
        discussion=engine.recommend(user_id, "discussion", prefs, 50),
        community=engine.recommend(user_id, "community", prefs, 20),
    )


@recommend_router.post("/mealplan/{user_id}", response_model=MealPlanResponse)
def generate_mealplan_for_user(
    user_id: str,
    payload: MealPlanRequest,
):
    engine = dependencies.engine
    engine.refresh_models()
    prefs = get_user_preferences(user_id)
    exclude_ids = get_mealplan_exclude_ids(user_id, payload.date)

    raw_session = (
        fetch_session_data(user_id=user_id, date=str(payload.date))
        if payload.target_recipe_id
        else None
    )

    session_data = []
    if isinstance(raw_session, list):
        for s in raw_session:
            if isinstance(s, str):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        session_data.extend(parsed)
                    elif isinstance(parsed, dict):
                        session_data.append(parsed)
                except json.JSONDecodeError:
                    continue
            elif isinstance(s, dict):
                session_data.append(s)

    meals: List[MealItem] = []

    for mealtime in payload.mealtime:
        result = engine.generate_mealplan(
            mealtime=mealtime,
            user_id=user_id,
            prefs=prefs,
            exclude_ids=exclude_ids,
            session_data=session_data,
            target_recipe_id=payload.target_recipe_id,
        )

        recipes_raw = result["recipes"]
        session_data = result["session_data"]
        recipe_items = [
            RecipeItem(
                id=recipe["recipe_id"],
                title=recipe.get("title", ""),
                image=recipe.get("image", ""),
            )
            for recipe in recipes_raw
        ]

        meals.append(
            MealItem(
                mealtime=mealtime,
                recipes=recipe_items,
                session=json.dumps(session_data),
            )
        )

    return MealPlanResponse(
        user_id=user_id,
        date=payload.date,
        meals=meals,
    )
