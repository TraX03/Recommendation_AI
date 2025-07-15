from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from app import dependencies
from app.api.coldstart import coldstart_router
from app.api.feedback import feedback_router
from app.api.interaction import interaction_router
from app.api.recommendation import recommend_router
from app.engines.recommender import Recommender
from app.utils.background_tasks import start_background_tasks
from app.utils.data_loader import fetch_interaction_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initializing engine...")
    recipes_df = pd.read_pickle("precomputed/recipes.pkl")
    tips_df = pd.read_pickle("precomputed/tips.pkl")
    discussion_df = pd.read_pickle("precomputed/discussions.pkl")
    community_df = pd.read_pickle("precomputed/communities.pkl")
    inventory_df = pd.read_pickle("precomputed/inventory.pkl")
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
