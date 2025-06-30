import json

import pandas as pd
from fastapi import APIRouter, HTTPException

from app import dependencies
from app.constants import RECOMMENDATION_DATA_COLLECTION_ID, USERS_COLLECTION_ID
from app.models.response_models import PostList
from app.services.coldstart_service import (
    generate_post_coldstart,
    generate_recipe_coldstart,
)
from app.utils.appwrite_client import create_or_update_document, get_document_by_id

coldstart_router = APIRouter()


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


def to_post_list(df: pd.DataFrame, id_col: str = "post_id") -> PostList:
    return PostList(
        post_ids=df[id_col].tolist(),
        titles=df["title"].tolist(),
        images=df["image"].tolist(),
        author_ids=df["author_id"].tolist() if "author_id" in df.columns else [],
    )


@coldstart_router.post("/coldstart/{user_id}", response_model=PostList)
def cold_start_from_user(user_id: str) -> PostList:
    prefs = get_user_preferences(user_id)

    recipe_df = generate_recipe_coldstart(
        user_prefs=prefs,
        recipes_df=dependencies.recipes_df,
        tfidf_matrix=dependencies.tfidf_matrix,
        tfidf_vectorizer=dependencies.tfidf_vectorizer,
        max_recs=10,
    )

    suggestions = {
        "tip": generate_post_coldstart(
            user_prefs=prefs,
            df=dependencies.engine.tips_df,
            tfidf_vectorizer=dependencies.tfidf_vectorizer,
            max_recs=10,
        )["post_id"].tolist(),
        "discussion": generate_post_coldstart(
            user_prefs=prefs,
            df=dependencies.engine.discussions_df,
            tfidf_vectorizer=dependencies.tfidf_vectorizer,
            max_recs=10,
        )["post_id"].tolist(),
        "community": generate_post_coldstart(
            user_prefs=prefs,
            df=dependencies.engine.communities_df,
            tfidf_vectorizer=dependencies.tfidf_vectorizer,
            id_column="community_id",
            max_recs=10,
        )["community_id"].tolist(),
    }

    create_or_update_document(
        collection_id=RECOMMENDATION_DATA_COLLECTION_ID,
        document_id=user_id,
        data={
            "user_id": user_id,
            "onboarding_suggestions": json.dumps(suggestions),
        },
    )

    return to_post_list(recipe_df, id_col="recipe_id")
