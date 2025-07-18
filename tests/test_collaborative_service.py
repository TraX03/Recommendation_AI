from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.services.collaborative_service import CollaborativeService


@pytest.fixture
def interaction_data():
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=1)
    return pd.DataFrame(
        [
            {
                "user_id": "user1",
                "item_id": "itemA",
                "item_type": "recipe",
                "type": "like",
                "score": None,
            },
            {
                "user_id": "user1",
                "item_id": "itemB",
                "item_type": "recipe",
                "type": "rating",
                "score": 8.0,
            },
            {
                "user_id": "user2",
                "item_id": "itemB",
                "item_type": "recipe",
                "type": "view",
                "timestamps": [past.isoformat(), now.isoformat()],
            },
        ]
    )


def test_prepare_interactions_structure(interaction_data):
    service = CollaborativeService()
    result = service._prepare_interactions(interaction_data)

    assert isinstance(result, pd.DataFrame)
    assert {"user_id", "item_id", "item_type", "type", "score"}.issubset(result.columns)
    assert "view" in result["type"].values
    assert "rating" in result["type"].values
    assert "like" in result["type"].values


def test_normalize_score_rating():
    service = CollaborativeService()
    row = {"type": "rating", "score": 7}
    score = service._normalize_score(row)
    assert score == 0.7


def test_normalize_score_missing_score():
    service = CollaborativeService()
    row = {"type": "bookmark", "score": None}
    score = service._normalize_score(row)
    assert isinstance(score, float)


def test_generate_view_scores_recent():
    service = CollaborativeService()

    now = datetime.now(timezone.utc)
    df = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "item_type": "recipe",
                "type": "view",
                "timestamps": [now.isoformat(), now.isoformat()],
            }
        ]
    )
    result = service._generate_view_scores(df)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "score" in result.columns
    assert result.iloc[0]["score"] <= 0.6
    assert result.iloc[0]["type"] == "view"


def test_build_cf_matrix_valid(interaction_data):
    service = CollaborativeService()
    processed = service._prepare_interactions(interaction_data)

    model = service._build_cf_matrix(processed, "recipe")

    assert isinstance(model, dict)
    assert "matrix" in model
    assert "item_map" in model
    assert "item_idx_to_id" in model
    assert model["matrix"].shape[0] == len(model["item_map"])


def test_build_cf_matrix_empty():
    service = CollaborativeService()
    df = pd.DataFrame(columns=["user_id", "item_id", "item_type", "type", "score"])
    model = service._build_cf_matrix(df, "recipe")
    assert model == {}


def test_prepare_cf_models_end_to_end(interaction_data):
    service = CollaborativeService()
    models = service.prepare_cf_models(interaction_data, ["recipe"])

    assert isinstance(models, dict)
    assert "recipe" in models
    assert "matrix" in models["recipe"]
