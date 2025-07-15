from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import dependencies
from app.api.coldstart import coldstart_router
from app.api.feedback import feedback_router
from app.api.interaction import interaction_router
from app.api.recommendation import recommend_router
from app.engines.recommender import Recommender
from app.utils.background_tasks import start_background_tasks
from app.utils.data_loader import (
    fetch_community_data,
    fetch_interaction_data,
    fetch_inventory_data,
    fetch_post_data,
    fetch_recipe_data,
)
from app.utils.embedding_utils import load_or_embed


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initializing engine...")
    recipes_df = load_or_embed("recipes", fetch_recipe_data)
    tips_df, discussion_df = fetch_post_data()
    tips_df = load_or_embed("tips", lambda: tips_df)
    discussion_df = load_or_embed("discussions", lambda: discussion_df)
    community_df = load_or_embed("communities", fetch_community_data)
    inventory_df = load_or_embed("inventory", fetch_inventory_data, text_column="name")
    interactions_df = fetch_interaction_data()

    engine = Recommender(
        recipes_df=recipes_df,
        interactions_df=interactions_df,
        tips_df=tips_df,
        discussions_df=discussion_df,
        communities_df=community_df,
        inventory_df=inventory_df,
    )

    dependencies.engine = engine

    await start_background_tasks()

    print("[Startup] Engine ready. Background tasks running.")
    yield

    print("[Shutdown] Cleaning up.")
    dependencies.engine = None


app = FastAPI(lifespan=lifespan)

app.include_router(recommend_router, prefix="/recommendation")
app.include_router(coldstart_router, prefix="/onboarding")
app.include_router(interaction_router, prefix="/interaction")
app.include_router(feedback_router, prefix="/feedback")


@app.get("/")
def root():
    return {"message": "Recommendation API is running."}
