import json
from typing import Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException

from app import dependencies
from app.constants import RECOMMENDATION_DATA_COLLECTION_ID, USERS_COLLECTION_ID
from app.models.response_models import PostList, RecommendationResponse
from app.services.recommendation_service import ID_COLUMN_MAP, RecommendationEngine
from app.utils.appwrite_client import create_or_update_document, get_document_by_id
from app.utils.filtering_utils import filter_avoid_ingredients, filter_recent_seen

recommend_router = APIRouter()


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


def get_weights(seed_id: str, engine: RecommendationEngine) -> Tuple[float, float]:
    return engine.adaptive_weights(seed_id) if engine.data_sufficient else (0.6, 0.4)


@recommend_router.post("/homeFeed/{user_id}", response_model=RecommendationResponse)
def recommend_for_home(user_id: str) -> RecommendationResponse:
    engine = dependencies.engine

    prefs = get_user_preferences(user_id)
    interactions = engine.interactions_df.query("user_id == @user_id").copy()
    interactions["created_at"] = pd.to_datetime(
        interactions["created_at"], errors="coerce"
    )
    interactions = interactions.sort_values(by="created_at", ascending=False)

    def generate_recommendations(content_type: str, max_count: int) -> PostList:
        id_col = ID_COLUMN_MAP[content_type]

        trending_df = engine.get_trending_items(content_type, n=10)
        trending = trending_df[id_col].tolist()

        recent = (
            interactions[interactions["item_type"] == content_type]["item_id"]
            .dropna()
            .unique()
            .tolist()[:10]
        )
        trending_extras = [r for r in trending if r not in recent][:2]

        seeds = (recent + trending_extras) if (recent + trending_extras) else trending
        seed_threshold = 5 if content_type == "community" else 10

        if len(seeds) < seed_threshold:
            coldstart_key = {
                "tip": "tip",
                "discussion": "discussion",
                "community": "community",
            }.get(content_type)

            try:
                doc = get_document_by_id(RECOMMENDATION_DATA_COLLECTION_ID, user_id)
                onboarding_suggestions = json.loads(
                    doc.get("onboarding_suggestions", "{}")
                )
            except Exception:
                onboarding_suggestions = {}

            onboarding_ids = (
                onboarding_suggestions.get(coldstart_key, []) if coldstart_key else []
            )

            if not seeds:
                seeds = onboarding_ids[:seed_threshold]
            else:
                extra_needed = seed_threshold - len(seeds)
                top_up = [cid for cid in onboarding_ids if cid not in seeds][
                    :extra_needed
                ]
                seeds += top_up

        all_recommendations = [
            engine.get_hybrid_recommendations(
                seed_id,
                content_type,
                *get_weights(seed_id, engine),
                top_k=100,
                sample_n=max_count,
            )
            for seed_id in seeds
        ]

        combined_df = pd.concat(all_recommendations).drop_duplicates(subset=id_col)
        filtered = filter_recent_seen(combined_df, user_id, id_col)

        if content_type == "recipe":
            filtered = filter_avoid_ingredients(filtered, prefs["avoid_ingredients"])

        filtered = filtered.head(max_count)
        last_ids = filtered.head(60)[id_col].tolist()

        create_or_update_document(
            collection_id=RECOMMENDATION_DATA_COLLECTION_ID,
            document_id=user_id,
            data={
                "user_id": user_id,
                "last_recommendations": last_ids,
            },
        )

        return PostList(
            post_ids=filtered[id_col].tolist(),
            titles=filtered["title"].tolist(),
            images=filtered["image"].tolist(),
            author_ids=filtered["author_id"].tolist(),
        )

    return RecommendationResponse(
        recipe=generate_recommendations("recipe", 100),
        tip=generate_recommendations("tip", 100),
        discussion=generate_recommendations("discussion", 100),
        community=generate_recommendations("community", 30),
    )
