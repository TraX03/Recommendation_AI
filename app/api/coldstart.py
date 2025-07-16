import json

import pandas as pd
from fastapi import APIRouter

from app import dependencies
from app.constants import RECOMMENDATION_DATA_COLLECTION_ID
from app.models.schemas import PostList
from app.utils.appwrite_client import create_or_update_document
from app.utils.data_loader import get_user_preferences

coldstart_router = APIRouter()


def to_post_list(df: pd.DataFrame, id_col: str = "post_id") -> PostList:
    return PostList(
        post_ids=df[id_col].tolist(),
        titles=df["title"].tolist(),
        images=df["image"].tolist(),
        author_ids=df["author_id"].tolist() if "author_id" in df.columns else [],
    )


@coldstart_router.post("/coldstart/{user_id}", response_model=PostList)
def cold_start_from_user(user_id: str) -> PostList:
    engine = dependencies.engine
    engine.refresh_models()
    prefs = get_user_preferences(user_id)
    result = engine.cold_start(user_prefs=prefs)

    recipe_df = result["recipe_df"]
    suggestions = result["suggestions"]

    create_or_update_document(
        collection_id=RECOMMENDATION_DATA_COLLECTION_ID,
        document_id=user_id,
        data={
            "user_id": user_id,
            "onboarding_suggestions": json.dumps(suggestions),
        },
    )

    return to_post_list(recipe_df, id_col="recipe_id")
