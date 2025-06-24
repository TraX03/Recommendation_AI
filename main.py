import json
import os
from typing import List

import numpy as np
import pandas as pd
from appwrite.client import Client
from appwrite.query import Query
from appwrite.services.databases import Databases
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale

load_dotenv()
app = FastAPI()


# Fetch data
def fetch_recipe_data():
    client = Client()
    client.set_endpoint("https://cloud.appwrite.io/v1").set_project(
        os.environ.get("APPWRITE_PROJECT_ID")
    ).set_key(os.environ.get("APPWRITE_API_KEY"))

    database = Databases(client)

    documents = []
    limit = 100
    offset = 0

    while True:
        result = database.list_documents(
            database_id=os.environ.get("APPWRITE_DATABASE_ID"),
            collection_id=os.environ.get("APPWRITE_RECIPES_COLLECTION_ID"),
            queries=[Query.limit(limit), Query.offset(offset)],
        )
        docs = result["documents"]
        if not docs:
            break
        documents.extend(docs)
        offset += limit

    processed_data = []
    for doc in documents:
        try:
            ingredients = [
                json.loads(item)["name"] for item in doc.get("ingredients", [])
            ]
        except Exception:
            ingredients = []

        tags = doc.get("tags", [])
        mealtime = doc.get("mealtime", [])
        cuisine = doc.get("area", "") or doc.get("category", "") or ""

        combined_text = " ".join(ingredients + tags + mealtime + [cuisine])

        processed_data.append(
            {
                "recipe_id": doc["$id"],
                "title": doc.get("title", "Untitled"),
                "combined_text": combined_text,
            }
        )

    return pd.DataFrame(processed_data)


# Load data
cbf_df = fetch_recipe_data()
cf_df = pd.read_excel("user_interactions_mock.xlsx")


# Content-based filtering
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(cbf_df["combined_text"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(cbf_df.index, index=cbf_df["recipe_id"]).drop_duplicates()


# Collaborative filtering
INTERACTION_WEIGHTS = {
    "coldStart": {"like": 1.0, "neutral": 0.5, "dislike": 0.0},
    "like": 1.0,
    "bookmark": 0.5,
    "view": 0.2,
}


def compute_score(row):
    if row["type"] == "coldStart":
        return INTERACTION_WEIGHTS["coldStart"].get(row["value"], 0.0)
    return INTERACTION_WEIGHTS.get(row["type"], 0.0)


cf_df["score"] = cf_df.apply(compute_score, axis=1)

user_ids = cf_df["userid"].astype(str).unique()
recipe_ids = cf_df["recipeid"].astype(str).unique()

user_map = {uid: idx for idx, uid in enumerate(user_ids)}
recipe_map = {rid: idx for idx, rid in enumerate(recipe_ids)}

cf_df["user_idx"] = cf_df["userid"].astype(str).map(user_map)
cf_df["recipe_idx"] = cf_df["recipeid"].astype(str).map(recipe_map)

interaction_matrix = csr_matrix(
    (cf_df["score"], (cf_df["user_idx"], cf_df["recipe_idx"])),
    shape=(len(user_map), len(recipe_map)),
)

recipe_sim_matrix = cosine_similarity(interaction_matrix.T)


# Hybrid
def adaptive_weights(recipe_id):
    rid = str(recipe_id)
    interaction_count = cf_df[cf_df["recipeid"] == rid].shape[0]
    total_interactions = cf_df.shape[0]
    ratio = interaction_count / total_interactions if total_interactions > 0 else 0
    cf_weight = min(0.8, ratio + 0.1)
    cbf_weight = 1.0 - cf_weight
    return cbf_weight, cf_weight


def get_hybrid_recommendations(recipe_id, cbf_weight=0.5, cf_weight=0.5, top_n=5):
    cbf_idx = indices[recipe_id]
    cbf_scores = cosine_sim[cbf_idx]
    cf_scores_full = np.zeros_like(cbf_scores)

    try:
        cf_idx = recipe_map[str(recipe_id)]
        cf_scores_partial = recipe_sim_matrix[cf_idx]
        for rid_str, partial_idx in recipe_map.items():
            if rid_str in indices:
                full_idx = indices[rid_str]
                cf_scores_full[full_idx] = cf_scores_partial[partial_idx]
    except KeyError:
        pass

    cbf_scores_norm = minmax_scale(cbf_scores)
    cf_scores_norm = minmax_scale(cf_scores_full)
    hybrid_scores = cbf_weight * cbf_scores_norm + cf_weight * cf_scores_norm

    sim_scores = list(enumerate(hybrid_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != cbf_idx][:top_n]
    hybrid_indices = [i[0] for i in sim_scores]
    return cbf_df.iloc[hybrid_indices][["recipe_id", "title"]]


# API models
class RecipeRequest(BaseModel):
    recipe_id: str
    top_n: int = 5


class RecommendationResponse(BaseModel):
    recipe_ids: List[str]
    titles: List[str]


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(data: RecipeRequest):
    recipe_id = data.recipe_id

    if recipe_id not in indices:
        raise HTTPException(status_code=404, detail="Recipe ID not found")

    # cbf_weight, cf_weight = adaptive_weights(recipe_id)
    try:
        recommendations = get_hybrid_recommendations(
            recipe_id, cbf_weight=0.6, cf_weight=0.4, top=data.top_n
        )
        return {
            "recipe_ids": recommendations["recipe_id"].tolist(),
            "titles": recommendations["title"].tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
