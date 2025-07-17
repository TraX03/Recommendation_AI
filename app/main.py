from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import dependencies
from app.api.coldstart import coldstart_router
from app.api.feedback import feedback_router
from app.api.interaction import interaction_router
from app.api.recommendation import recommend_router
from app.engines.recommender import Recommender
from app.utils.background_tasks import start_background_tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initializing engine...")
    engine = Recommender()
    await start_background_tasks()

    dependencies.engine = engine

    yield
    dependencies.engine = None
    dependencies.embedding_model = None


app = FastAPI(lifespan=lifespan)

app.include_router(recommend_router, prefix="/recommendation")
app.include_router(coldstart_router, prefix="/onboarding")
app.include_router(interaction_router, prefix="/interaction")
app.include_router(feedback_router, prefix="/feedback")


@app.get("/")
def root():
    return {"message": "Recommendation API is running."}
