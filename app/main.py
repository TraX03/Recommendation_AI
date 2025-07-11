from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import dependencies
from app.api.coldstart import coldstart_router
from app.api.log_view import log_view_router
from app.api.recommend import recommend_router
from app.services.recommendation_service import HybridRecommender
from app.utils.embedding_utils import load_or_embed
from app.utils.load_utils import (
    fetch_community_data,
    fetch_interaction_data,
    fetch_post_data,
    fetch_recipe_data,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    recipes_df = load_or_embed("recipes", fetch_recipe_data)
    tips_df, discussion_df = fetch_post_data()
    tips_df = load_or_embed("tips", lambda: tips_df)
    discussion_df = load_or_embed("discussions", lambda: discussion_df)
    community_df = load_or_embed("communities", fetch_community_data)
    interactions_df = fetch_interaction_data()

    engine = HybridRecommender(
        recipes_df=recipes_df,
        interactions_df=interactions_df,
        tips_df=tips_df,
        discussions_df=discussion_df,
        communities_df=community_df,
    )

    dependencies.engine = engine

    yield

    dependencies.engine = None


app = FastAPI(lifespan=lifespan)

app.include_router(recommend_router, prefix="/recommendation")
app.include_router(coldstart_router, prefix="/onboarding")
app.include_router(log_view_router, prefix="/interactions")


@app.get("/")
def root():
    return {"message": "Recommendation API is running."}
