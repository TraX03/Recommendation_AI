from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.services.hybrid_recommendation_service import HybridRecommendationService


@pytest.fixture
def realistic_data():
    recipe_df = pd.DataFrame(
        {
            "recipe_id": ["r1", "r2", "r3", "r4"],
            "mealtime": [["lunch"], ["dinner"], ["all"], ["lunch", "dinner"]],
            "ingredients": [
                ["chicken", "rice"],
                ["tofu"],
                ["broccoli"],
                ["beef", "rice"],
            ],
            "embedding": [np.random.rand(10) for _ in range(4)],
            "title": [
                "Chicken Rice",
                "Tofu Stir Fry",
                "Broccoli Bowl",
                "Beef Rice Bowl",
            ],
            "image": ["img1", "img2", "img3", "img4"],
            "author_id": ["a1", "a2", "a3", "a4"],
            "combined_text": [
                "chicken rice recipe lunch",
                "tofu stir fry recipe dinner",
                "broccoli bowl recipe all",
                "beef rice bowl recipe lunch dinner",
            ],
        }
    )

    interactions_df = pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2"],
            "item_type": ["recipe", "recipe", "recipe"],
            "item_id": ["r1", "r2", "r3"],
            "type": ["like", "bookmark", "view"],
        }
    )

    inventory_df = pd.DataFrame(
        {
            "name": ["chicken", "tofu", "rice"],
            "expiries": [["2025-07-20"], ["2025-07-18"], ["2025-07-19"]],
        }
    )

    sim_model = {
        "indices": {"r1": 0, "r2": 1, "r3": 2, "r4": 3},
        "cosine_sim": np.identity(4),
    }

    cf_matrix = {
        "item_map": {"r1": 0, "r2": 1, "r3": 2, "r4": 3},
        "matrix": np.identity(4),
    }

    return recipe_df, interactions_df, inventory_df, sim_model, cf_matrix


def get_mock_service(recipe_df_length=4):
    build_tfidf_model = MagicMock()
    build_tfidf_model.return_value = {
        "vectorizer": MagicMock(),
        "tfidf_matrix": MagicMock(),
        "indices": pd.Series(dtype=int),
    }

    build_user_profile_vector = MagicMock()
    build_user_profile_vector.return_value = np.ones(recipe_df_length) * 0.5

    return HybridRecommendationService(
        build_tfidf_model=build_tfidf_model,
        build_user_profile_vector=build_user_profile_vector,
    )


def test_hybrid_generate_mealplan_integration(realistic_data):
    (
        recipe_df,
        interactions_df,
        inventory_df,
        sim_model,
        cf_matrix,
    ) = realistic_data

    service = get_mock_service()

    result = service.generate_mealplan(
        mealtime="lunch",
        config={"dishCount": 2, "staples": "rice"},
        avoid_ingredients=["beef"],
        region_pref=["indian", "malaysian"],
        diet=["halal"],
        recipe_df=recipe_df,
        sim_model=sim_model,
        cf_matrix=cf_matrix,
        inventory_df=inventory_df,
        interactions_df=interactions_df,
    )

    assert isinstance(result, dict)
    assert "recipes" in result
    assert "session_data" in result

    recipes = result["recipes"]
    session_data = result["session_data"]

    assert isinstance(recipes, list)
    assert len(recipes) <= 2

    for r in recipes:
        ingredients = recipe_df.loc[
            recipe_df["recipe_id"] == r["recipe_id"], "ingredients"
        ].values[0]
        assert "beef" not in ingredients

    assert len(session_data) == len(recipes)

    output_ids = {r["recipe_id"] for r in recipes}
    input_ids = set(recipe_df["recipe_id"].values)
    assert output_ids.issubset(input_ids)


def test_hybrid_mealplan_with_empty_inventory(realistic_data):
    recipe_df, interactions_df, _, sim_model, cf_matrix = realistic_data
    inventory_df_empty = pd.DataFrame(columns=["name", "expiries"])

    service = get_mock_service()

    result = service.generate_mealplan(
        mealtime="dinner",
        config={"dishCount": 1, "staples": ""},
        avoid_ingredients=[],
        region_pref=[],
        diet=[],
        recipe_df=recipe_df,
        sim_model=sim_model,
        cf_matrix=cf_matrix,
        inventory_df=inventory_df_empty,
        interactions_df=interactions_df,
    )

    assert "recipes" in result
    assert len(result["recipes"]) <= 1


def test_hybrid_mealplan_avoids_allergen(realistic_data):
    recipe_df, interactions_df, inventory_df, sim_model, cf_matrix = realistic_data

    service = get_mock_service()

    avoid = ["chicken", "tofu"]

    result = service.generate_mealplan(
        mealtime="lunch",
        config={"dishCount": 3},
        avoid_ingredients=avoid,
        region_pref=[],
        diet=[],
        recipe_df=recipe_df,
        sim_model=sim_model,
        cf_matrix=cf_matrix,
        inventory_df=inventory_df,
        interactions_df=interactions_df,
    )

    for r in result["recipes"]:
        ingredients = recipe_df.loc[
            recipe_df["recipe_id"] == r["recipe_id"], "ingredients"
        ].values[0]
        assert not any(i in avoid for i in ingredients)
