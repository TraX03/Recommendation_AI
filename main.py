import json
import os
import random
import shelve
from typing import List, Optional

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

# Setup appwrite
client = Client()
client.set_endpoint("https://cloud.appwrite.io/v1").set_project(
    os.environ.get("APPWRITE_PROJECT_ID")
).set_key(os.environ.get("APPWRITE_API_KEY"))

database = Databases(client)


# Fetch data
def fetch_recipe_data():
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
        description = doc.get("description", "")
        if description is None:
            description = ""
        instructions = [
            step.get("text", "")
            for step in doc.get("instructions", [])
            if isinstance(step, dict)
        ]

        combined_text = " ".join(
            ingredients + tags + mealtime + [cuisine, description] + instructions
        )

        image_list = doc.get("image", [])
        first_image = image_list[0] if image_list else ""

        processed_data.append(
            {
                "recipe_id": doc["$id"],
                "title": doc.get("title"),
                "combined_text": combined_text,
                "image": first_image,
                "author_id": doc.get("author_id"),
            }
        )

    return pd.DataFrame(processed_data)


def fetch_interaction_data():
    documents = []
    limit = 100
    offset = 0

    while True:
        result = database.list_documents(
            database_id=os.environ.get("APPWRITE_DATABASE_ID"),
            collection_id=os.environ.get("APPWRITE_INTERACTIONS_COLLECTION_ID"),
            queries=[Query.limit(limit), Query.offset(offset)],
        )
        docs = result["documents"]
        if not docs:
            break
        documents.extend(docs)
        offset += limit

    processed_data = []
    for doc in documents:
        processed_data.append(
            {
                "interaction_id": doc["$id"],
                "user_id": doc.get("user_id"),
                "recipe_id": doc.get("item_id"),
                "type": doc.get("type"),
                "value": doc.get("value"),
                "created_at": doc.get("created_at"),
            }
        )

    df_appwrite = pd.DataFrame(processed_data)

    # If less than 1000 interactions, append mock data
    if len(df_appwrite) < 1000:
        mock_df = pd.read_excel("mockData/user_interactions.xlsx")
        df_appwrite = pd.concat([df_appwrite, mock_df], ignore_index=True)

    return df_appwrite


def get_user_preferences(user_id: str):
    try:
        user_doc = database.get_document(
            database_id=os.environ["APPWRITE_DATABASE_ID"],
            collection_id=os.environ["APPWRITE_USERS_COLLECTION_ID"],
            document_id=user_id,
        )
    except Exception:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "avoid_ingredients": user_doc.get("avoid_ingredients", []),
        "diet": user_doc.get("diet", []),
        "region_pref": user_doc.get("region_pref", []),
    }


# Load data
recipes_df = fetch_recipe_data()
interactions_df = fetch_interaction_data()


# Content-based filtering
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(recipes_df["combined_text"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(recipes_df.index, index=recipes_df["recipe_id"]).drop_duplicates()


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


interactions_df["score"] = interactions_df.apply(compute_score, axis=1)

user_ids = interactions_df["user_id"].astype(str).unique()
recipe_ids = interactions_df["recipe_id"].astype(str).unique()

user_map = {uid: idx for idx, uid in enumerate(user_ids)}
recipe_map = {rid: idx for idx, rid in enumerate(recipe_ids)}

interactions_df["user_idx"] = interactions_df["user_id"].astype(str).map(user_map)
interactions_df["recipe_idx"] = interactions_df["recipe_id"].astype(str).map(recipe_map)

interaction_matrix = csr_matrix(
    (
        interactions_df["score"],
        (interactions_df["user_idx"], interactions_df["recipe_idx"]),
    ),
    shape=(len(user_map), len(recipe_map)),
)

recipe_sim_matrix = cosine_similarity(interaction_matrix.T)


# Hybrid
def adaptive_weights(recipe_id):
    rid = str(recipe_id)
    interaction_count = interactions_df[interactions_df["recipe_id"] == rid].shape[0]
    total_interactions = interactions_df.shape[0]
    ratio = interaction_count / total_interactions if total_interactions > 0 else 0
    cf_weight = min(0.8, ratio + 0.1)
    cbf_weight = 1.0 - cf_weight
    return cbf_weight, cf_weight


def get_hybrid_recommendations(
    recipe_id, cbf_weight=0.6, cf_weight=0.4, top_k=100, sample_n=20
):
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
    hybrid_scores = (
        cbf_weight * cbf_scores_norm
        + cf_weight * cf_scores_norm
        + np.random.normal(0, 0.01, size=len(cbf_scores))  # add Gaussian noise
    )

    sim_scores = list(enumerate(hybrid_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != cbf_idx][:top_k]

    sampled = random.sample(sim_scores, min(sample_n, len(sim_scores)))
    hybrid_indices = [i[0] for i in sampled]

    return recipes_df.iloc[hybrid_indices][
        ["recipe_id", "title", "image", "author_id", "combined_text"]
    ]


# Support functions
def get_trending_recipes(n=20):
    top = (
        interactions_df[interactions_df["type"] == "like"]
        .groupby("recipe_id")
        .size()
        .sort_values(ascending=False)
        .head(n)
        .index
    )
    return recipes_df[recipes_df["recipe_id"].isin(top)][["recipe_id", "title"]]


def save_last_recommendations(user_id: str, recipe_ids: list[str]):
    with shelve.open("last_recommendations.db") as db:
        previous_rounds = db.get(user_id, [])
        updated = previous_rounds[-1:] + [recipe_ids]
        db[user_id] = updated[-2:]


def get_last_recommendations(user_id: str) -> list[list[str]]:
    with shelve.open("last_recommendations.db") as db:
        return db.get(user_id, [])


def filter_recent_seen(combined_df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    last_two_rounds = get_last_recommendations(user_id)

    seen_strict = set()
    for round_recipes in last_two_rounds:
        seen_strict.update(round_recipes[:60])  # only first 70 from each round

    filtered_df = combined_df[~combined_df["recipe_id"].isin(seen_strict)]

    return filtered_df


def filter_avoid_ingredients(
    df: pd.DataFrame, avoid_ingredients: List[str]
) -> pd.DataFrame:
    if not avoid_ingredients:
        return df

    avoid_ingredients_lower = [a.lower() for a in avoid_ingredients]

    def contains_avoid(text: str):
        return any(a in text.lower() for a in avoid_ingredients_lower)

    return df[~df["combined_text"].apply(contains_avoid)]


# API models
class RecommendationResponse(BaseModel):
    recipe_ids: List[str]
    titles: List[str]
    images: List[str]
    author_ids: Optional[List[str]] = None


@app.post("/coldstart/{user_id}", response_model=RecommendationResponse)
def cold_start_from_user(user_id: str):
    prefs = get_user_preferences(user_id)

    # Step 1: Create synthetic user profile
    user_profile_text = " ".join(prefs["diet"] + prefs["region_pref"])
    user_vector = tfidf.transform([user_profile_text])

    # Step 2: Compute similarity scores
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    ranked_indices = similarities.argsort()[::-1]

    # Step 3: Build DataFrame of ranked recipes
    ranked_df = recipes_df.iloc[ranked_indices].copy()
    ranked_df["similarity"] = similarities[ranked_indices]

    # Step 4: Strict filtering by avoid ingredients
    filtered_df = filter_avoid_ingredients(ranked_df, prefs["avoid_ingredients"])

    # Step 5: Separate preferred and fallback based on match with diet/region
    def is_preferred(row):
        text = row["combined_text"].lower()
        return all(
            [
                any(d.lower() in text for d in prefs["diet"])
                if prefs["diet"]
                else True,
                any(r.lower() in text for r in prefs["region_pref"])
                if prefs["region_pref"]
                else True,
            ]
        )

    preferred = filtered_df[filtered_df.apply(is_preferred, axis=1)]
    fallback = filtered_df[~filtered_df.apply(is_preferred, axis=1)]

    final_df = (
        pd.concat([preferred.head(10), fallback.head(10)]).drop_duplicates().head(10)
    )
    sampled = final_df.sample(frac=1)

    return {
        "recipe_ids": sampled["recipe_id"].tolist(),
        "titles": sampled["title"].tolist(),
        "images": sampled["image"].tolist(),
    }


@app.post("/homeFeed/{user_id}", response_model=RecommendationResponse)
def recommend_for_home(user_id: str):
    prefs = get_user_preferences(user_id)

    # Step 1: Choose seed recipes
    user_interactions = interactions_df[interactions_df["user_id"] == user_id]
    user_interactions = user_interactions.sort_values(by="created_at", ascending=False)
    recent_recipes = user_interactions["recipe_id"].dropna().unique().tolist()[:10]
    trending_recipes = get_trending_recipes(n=10)["recipe_id"].tolist()
    trending_filtered = [r for r in trending_recipes if r not in recent_recipes][:2]
    seed_recipes = recent_recipes + trending_filtered

    if not seed_recipes:
        seed_recipes = get_trending_recipes(n=10)["recipe_id"].tolist()

    # Step 2: Get new recommendations
    results = []
    for rid in seed_recipes:
        cbf_weight, cf_weight = adaptive_weights(rid)
        recs = get_hybrid_recommendations(
            rid, cbf_weight, cf_weight, top_k=50, sample_n=20
        )
        results.append(recs)

    combined_df = pd.concat(results).drop_duplicates(subset="recipe_id")

    # Step 3: Filter seen recipes and avoid ingredients
    fresh_df = filter_recent_seen(combined_df, user_id)
    filtered_df = filter_avoid_ingredients(fresh_df, prefs["avoid_ingredients"])

    # Step 4: Sample and return
    sampled_df = filtered_df.sample(n=min(100, len(filtered_df)))
    save_last_recommendations(user_id, sampled_df["recipe_id"].tolist())

    return {
        "recipe_ids": sampled_df["recipe_id"].tolist(),
        "titles": sampled_df["title"].tolist(),
        "images": sampled_df["image"].tolist(),
        "author_ids": sampled_df["author_id"].tolist(),
    }
