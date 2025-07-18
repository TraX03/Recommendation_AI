from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

inventory_df = pd.DataFrame(
    {
        "name": ["chicken", "tofu"],
        "expiries": [
            [datetime(2025, 7, 20, tzinfo=timezone.utc)],
            [datetime(2025, 7, 18, tzinfo=timezone.utc)],
        ],
    }
)


def dummy_build_tfidf_model(df, id_col="index"):
    return {
        "vectorizer": MagicMock(transform=lambda x: np.random.rand(1, len(df))),
        "tfidf_matrix": np.random.rand(len(df), 50),
    }


def dummy_user_profile_vector(user_interactions, data_map, prefs, tfidf_model):
    return np.random.rand(len(data_map["recipe"]))


@pytest.fixture
def recommendation_service():
    from app.services.hybrid_recommendation_service import HybridRecommendationService

    return HybridRecommendationService(
        build_tfidf_model=dummy_build_tfidf_model,
        build_user_profile_vector=dummy_user_profile_vector,
    )


@pytest.fixture
def dummy_data():
    recipes_df = pd.DataFrame(
        {
            "recipe_id": [f"r{i}" for i in range(5)],
            "title": [f"Recipe {i}" for i in range(5)],
            "image": [f"http://image.com/{i}.jpg" for i in range(5)],
            "author_id": [f"user{i}" for i in range(5)],
            "combined_text": ["ingredient1 ingredient2"] * 5,
        }
    )

    sim_model = {
        "indices": {f"r{i}": i for i in range(5)},
        "cosine_sim": np.random.rand(5, 5),
    }

    cf_model = {
        "item_map": {f"r{i}": i for i in range(5)},
        "matrix": np.random.rand(5, 5),
    }

    interactions_df = pd.DataFrame(
        {
            "user_id": ["test_user"] * 3,
            "item_id": ["r0", "r1", "r2"],
            "item_type": ["recipe"] * 3,
            "type": ["like", "like", "like"],
        }
    )

    return {
        "user_id": "test_user",
        "content_type": "recipe",
        "prefs": {
            "avoid_ingredients": [],
            "diet": [],
            "region_pref": [],
        },
        "interactions": interactions_df,
        "data_map": {"recipe": recipes_df},
        "sim_models": {"recipe": sim_model},
        "cf_models": {"recipe": cf_model},
        "max_count": 3,
    }


@patch("app.services.hybrid_recommendation_service.create_or_update_document")
@patch("app.services.hybrid_recommendation_service.get_inferred_tags")
@patch("app.services.hybrid_recommendation_service.choose_feed_fallback_action")
def test_generate_recommendations_without_fallback(
    mock_strategy, mock_infer_tags, mock_create_doc, recommendation_service, dummy_data
):
    dummy_data["max_count"] = 3
    result = recommendation_service.generate_recommendations(**dummy_data)

    assert isinstance(result.post_ids, list)
    assert isinstance(result.titles, list)
    assert len(result.post_ids) <= dummy_data["max_count"]
    assert all(isinstance(pid, str) for pid in result.post_ids)
    assert mock_infer_tags.call_count in [0, 1]

    mock_create_doc.assert_called_once()
    mock_strategy.assert_not_called()

    print("Post IDs (no fallback):", result.post_ids)


@patch("app.services.hybrid_recommendation_service.create_or_update_document")
@patch("app.services.hybrid_recommendation_service.get_inferred_tags")
@patch("app.services.hybrid_recommendation_service.choose_feed_fallback_action")
def test_generate_recommendations_with_fallback(
    mock_strategy, mock_infer_tags, mock_create_doc, recommendation_service, dummy_data
):
    dummy_data["max_count"] = 10

    result = recommendation_service.generate_recommendations(**dummy_data)

    assert isinstance(result.post_ids, list)
    assert isinstance(result.titles, list)
    assert len(result.post_ids) <= dummy_data["max_count"]
    assert all(isinstance(pid, str) for pid in result.post_ids)

    mock_create_doc.assert_called_once()
    mock_infer_tags.assert_called_once()
    mock_strategy.assert_called_once()

    print("Post IDs (with fallback):", result.post_ids)
