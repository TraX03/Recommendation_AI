from contextlib import asynccontextmanager

from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer

from app import dependencies
from app.api.coldstart import coldstart_router
from app.api.recommend import recommend_for_home, recommend_router
from app.services.recommendation_service import RecommendationEngine
from app.utils.loader_utils import fetch_interaction_data, fetch_recipe_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    recipes_df = fetch_recipe_data()
    interactions_df, used_mock_data = fetch_interaction_data()

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(recipes_df["combined_text"])

    engine = RecommendationEngine(recipes_df, interactions_df)
    engine.used_mock = used_mock_data
    engine.preprocess()

    dependencies.recipes_df = recipes_df
    dependencies.tfidf_matrix = tfidf_matrix
    dependencies.tfidf_vectorizer = tfidf

    recommend_for_home.__globals__["engine"] = engine

    yield

    dependencies.recipes_df = None
    dependencies.tfidf_matrix = None
    dependencies.tfidf_vectorizer = None


app = FastAPI(lifespan=lifespan)

app.include_router(recommend_router)
app.include_router(coldstart_router)
