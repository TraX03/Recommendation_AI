import json

import pandas as pd
from fastapi import APIRouter
from sklearn.feature_extraction.text import TfidfVectorizer

from app import dependencies
from app.constants import CONTENT_TYPE_MAP, RECOMMENDATION_DATA_COLLECTION_ID
from app.models.response_models import PostList
from app.services.coldstart_service import generate_coldstart
from app.services.recommendation_service import RecommendationEngine
from app.utils.appwrite_client import create_or_update_document
from app.utils.load_utils import get_user_preferences

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
    engine: RecommendationEngine = dependencies.engine
    prefs = get_user_preferences(user_id)
    tfidf = TfidfVectorizer(stop_words="english")

    recipe_df = generate_coldstart(
        user_prefs=prefs,
        df=engine.recipes_df,
        tfidf_vectorizer=tfidf,
        id_column="recipe_id",
        avoid_filter=True,
        fit_new=True,
        max_recs=10,
    )

    post_sources = {
        ctype: (config["id_col"], getattr(engine, config["attr"]))
        for ctype, config in CONTENT_TYPE_MAP.items()
        if ctype != "recipe"
    }

    suggestions = {
        key: generate_coldstart(
            user_prefs=prefs,
            df=source_df,
            tfidf_vectorizer=tfidf,
            id_column=id_col,
            fit_new=False,
            max_recs=10,
        )[id_col].tolist()
        for key, (id_col, source_df) in post_sources.items()
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
