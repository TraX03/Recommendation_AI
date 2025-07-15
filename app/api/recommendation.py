import json
from typing import List

import pandas as pd
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

recommend_router = APIRouter()


@recommend_router.post("/homeFeed/{user_id}", response_model=RecommendationResponse)
def recommend_for_home(user_id: str) -> RecommendationResponse:
    prefs = get_user_preferences(user_id)
    engine = dependencies.engine

    engine.start_session(user_id)

    interactions = engine.interactions_df

    if "user_id" in interactions.columns:
        interactions = interactions[interactions["user_id"] == user_id].copy()
    else:
        interactions = pd.DataFrame()

    interactions["created_at"] = pd.to_datetime(
        interactions["created_at"], errors="coerce"
    )
    interactions = interactions.sort_values(by="created_at", ascending=False)

    return RecommendationResponse(
        recipe=engine.recommend(user_id, "recipe", prefs, interactions, 100),
        tip=engine.recommend(user_id, "tip", prefs, interactions, 100),
        discussion=engine.recommend(user_id, "discussion", prefs, interactions, 100),
        community=engine.recommend(user_id, "community", prefs, interactions, 30),
    )


@recommend_router.post("/mealplan/{user_id}", response_model=MealPlanResponse)
def generate_mealplan_for_user(
    user_id: str,
    payload: MealPlanRequest,
):
    engine = dependencies.engine

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
            prefs=prefs,
            recipe_df=engine.data_map["recipe"],
            sim_model=engine.sim_models["recipe"],
            cf_matrix=engine.cf_models["recipe"],
            inventory_df=engine.inventory_df.query("user_id == @user_id"),
            interactions_df=engine.interactions_df.query("user_id == @user_id"),
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
