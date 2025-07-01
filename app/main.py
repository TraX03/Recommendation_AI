from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import dependencies
from app.api.coldstart import coldstart_router
from app.api.log_view import log_view_router
from app.api.recommend import recommend_router
from app.services.recommendation_service import RecommendationEngine
from app.utils.load_utils import (
    fetch_community_data,
    fetch_interaction_data,
    fetch_post_data,
    fetch_recipe_data,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    recipes_df = fetch_recipe_data()
    interactions_df, data_sufficient = fetch_interaction_data()
    tips_df, discussion_df = fetch_post_data()
    community_df = fetch_community_data()

    engine = RecommendationEngine(
        recipes_df=recipes_df,
        interactions_df=interactions_df,
        tips_df=tips_df,
        discussions_df=discussion_df,
        communities_df=community_df,
        data_sufficient=data_sufficient,
    )

    dependencies.engine = engine

    yield

    dependencies.engine = None


app = FastAPI(lifespan=lifespan)

app.include_router(recommend_router, prefix="/recommendation")
app.include_router(coldstart_router, prefix="/onboarding")
app.include_router(log_view_router, prefix="/interactions")
