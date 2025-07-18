from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_dataframes():
    recipe_df = pd.DataFrame(
        {
            "recipe_id": ["r1", "r2", "r3"],
            "mealtime": [["lunch"], ["dinner"], ["all"]],
            "ingredients": [["chicken", "rice"], ["tofu"], ["broccoli"]],
            "embedding": [[0.1] * 10, [0.2] * 10, [0.3] * 10],
            "title": ["Chicken Rice", "Tofu Stir Fry", "Broccoli Bowl"],
            "image": ["img1", "img2", "img3"],
            "author_id": ["a1", "a2", "a3"],
            "combined_text": [
                "chicken rice recipe lunch",
                "tofu stir fry recipe dinner",
                "broccoli bowl recipe all",
            ],
        }
    )

    interactions_df = pd.DataFrame(
        {
            "user_id": ["user_1"] * 3,
            "item_type": ["recipe"] * 3,
            "item_id": ["r1", "r2", "r3"],
            "type": ["like"] * 3,
        }
    )

    inventory_df = pd.DataFrame(
        {
            "name": ["chicken", "tofu"],
            "expiries": [["2025-07-20"], ["2025-07-18"]],
        }
    )

    return recipe_df, interactions_df, inventory_df


def test_generate_mealplan_basic(dummy_dataframes):
    from app.services.hybrid_recommendation_service import HybridRecommendationService

    build_tfidf_model = MagicMock(
        return_value={"vectorizer": MagicMock(), "tfidf_matrix": MagicMock()}
    )
    build_user_profile_vector = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))

    service = HybridRecommendationService(
        build_tfidf_model=build_tfidf_model,
        build_user_profile_vector=build_user_profile_vector,
    )

    recipe_df, interactions_df, inventory_df = dummy_dataframes

    result = service.generate_mealplan(
        mealtime="lunch",
        config={"dishCount": 1, "staples": "rice"},
        avoid_ingredients=["beef"],
        region_pref=["malaysian"],
        diet=["halal"],
        recipe_df=recipe_df,
        sim_model={"indices": {"r1": 0, "r2": 1, "r3": 2}, "cosine_sim": [[1] * 3] * 3},
        cf_matrix={"item_map": {"r1": 0, "r2": 1, "r3": 2}, "matrix": [[1] * 3] * 3},
        inventory_df=inventory_df,
        interactions_df=interactions_df,
    )

    assert "recipes" in result
    assert isinstance(result["recipes"], list)
    assert len(result["session_data"]) == len(result["recipes"])


def test_score_items(dummy_dataframes):
    from app.services.hybrid_recommendation_service import HybridRecommendationService

    build_tfidf_model = MagicMock(
        return_value={"vectorizer": MagicMock(), "tfidf_matrix": MagicMock()}
    )
    build_user_profile_vector = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))

    service = HybridRecommendationService(
        build_tfidf_model=build_tfidf_model,
        build_user_profile_vector=build_user_profile_vector,
    )

    recipe_df, interactions_df, _ = dummy_dataframes
    sim_model = {"indices": {"r1": 0}, "cosine_sim": [[0.9, 0.5, 0.1]]}
    cf_matrix = {"item_map": {"r1": 0}, "matrix": [[0.8, 0.4, 0.2]]}

    scored_df = service._score_items(
        item_id="r1",
        content_type="recipe",
        prefs={},
        sim_model=sim_model,
        cf_matrix=cf_matrix,
        content_df=recipe_df,
        user_interactions=interactions_df,
        top_k=2,
        sample_n=1,
    )

    assert not scored_df.empty
    assert "title" in scored_df.columns


def test_apply_inventory_strategy(dummy_dataframes):
    from app.services.hybrid_recommendation_service import HybridRecommendationService

    build_tfidf_model = MagicMock(
        return_value={"vectorizer": MagicMock(), "tfidf_matrix": MagicMock()}
    )
    build_user_profile_vector = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))

    service = HybridRecommendationService(
        build_tfidf_model=build_tfidf_model,
        build_user_profile_vector=build_user_profile_vector,
    )

    recipe_df, _, inventory_df = dummy_dataframes

    result = service._apply_inventory_strategy(
        candidates=recipe_df.copy(),
        inventory_action="boost_inventory_match",
        inventory_df=inventory_df,
        ingredient_expiry_map={},
        near_expiry_cutoff=pd.Timestamp("2025-07-19"),
    )

    assert "score" in result.columns
    assert isinstance(result, pd.DataFrame)


def test_regenerate_one_recipe(dummy_dataframes):
    from app.services.hybrid_recommendation_service import HybridRecommendationService

    build_tfidf_model = MagicMock(
        return_value={"vectorizer": MagicMock(), "tfidf_matrix": MagicMock()}
    )
    build_user_profile_vector = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))

    service = HybridRecommendationService(
        build_tfidf_model=build_tfidf_model,
        build_user_profile_vector=build_user_profile_vector,
    )

    recipe_df, _, _ = dummy_dataframes

    candidates = recipe_df.copy()
    session_data = [
        {
            "id": "r1",
            "timestamp": "2025-07-17T00:00:00",
            "mealtime": "lunch",
            "feedback": None,
            "action": "boost_inventory_match",
            "available_ingredients": ["chicken", "rice"],
            "source": "staple",
        }
    ]

    selected, session = service._regenerate_one_recipe(
        candidates=candidates,
        session_data=session_data,
        target_recipe_id="r1",
        config={"staples": "rice"},
        mealtime="lunch",
        inventory_action="boost_inventory_match",
        available_ingredients=["chicken", "rice"],
    )

    assert len(selected) <= 1
    assert session[0]["source"] == "staple"
