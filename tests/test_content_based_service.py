import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from app.services.content_based_service import ContentBasedService


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        [
            {
                "recipe_id": "item1",
                "combined_text": "spicy indian curry with lentils",
                "embedding": np.array([0.1, 0.2, 0.3]),
                "tags": ["indian", "spicy"],
                "title": "Lentil Curry",
                "description": "A delicious spicy lentil curry.",
            },
            {
                "recipe_id": "item2",
                "combined_text": "sweet japanese mochi dessert",
                "embedding": np.array([0.2, 0.1, 0.0]),
                "tags": ["japanese", "sweet"],
                "title": "Mochi",
                "description": "Soft rice dessert from Japan.",
            },
        ]
    )


@pytest.fixture
def content_map(sample_df):
    return {"recipe": sample_df}


def test_build_tfidf_model(sample_df):
    service = ContentBasedService()
    result = service.build_tfidf_model(sample_df, id_col="recipe_id")

    assert "tfidf_matrix" in result
    assert isinstance(result["tfidf_matrix"], csr_matrix)
    assert result["cosine_sim"].shape == (2, 2)
    assert "vectorizer" in result
    assert isinstance(result["indices"], pd.Series)


def test_build_embedding_similarity(sample_df):
    service = ContentBasedService()
    result = service._build_embedding_similarity(sample_df, id_col="recipe_id")

    assert isinstance(result, dict)
    assert result["cosine_sim"].shape == (2, 2)
    assert isinstance(result["indices"], pd.Series)


def test_build_similarity_model_with_embeddings(sample_df):
    service = ContentBasedService()
    result = service._build_similarity_model(sample_df, id_col="recipe_id")

    assert "cosine_sim" in result
    assert "vectorizer" in result
    assert "tfidf_matrix" in result
    assert "indices" in result


def test_build_similarity_model_without_embeddings(sample_df):
    sample_df = sample_df.drop(columns=["embedding"])
    service = ContentBasedService()
    result = service._build_similarity_model(sample_df, id_col="recipe_id")

    assert "cosine_sim" in result
    assert "vectorizer" in result
    assert "tfidf_matrix" in result
    assert "indices" in result


def test_prepare_cbf_models(content_map):
    service = ContentBasedService()
    result = service.prepare_cbf_models(content_map)

    assert isinstance(result, dict)
    assert "recipe" in result
    assert "cosine_sim" in result["recipe"]


def test_build_user_profile_vector_with_data(sample_df):
    service = ContentBasedService()

    user_interactions = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "item1", "type": "like"},
            {"user_id": "u1", "item_id": "item2", "type": "bookmark"},
        ]
    )
    tfidf_model = service.build_tfidf_model(sample_df, id_col="recipe_id")

    prefs = {"diet": ["vegetarian"], "region_pref": ["indian"], "tags": ["spicy"]}
    data_map = {"recipe": sample_df}

    vector = service.build_user_profile_vector(
        user_interactions, data_map, prefs, tfidf_model
    )

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (2,)
    assert np.any(vector > 0)


def test_build_user_profile_vector_with_empty_input(sample_df):
    service = ContentBasedService()

    user_interactions = pd.DataFrame(columns=["item_id", "type"])
    tfidf_model = service.build_tfidf_model(sample_df, id_col="recipe_id")
    prefs = {}
    data_map = {"recipe": sample_df}

    vector = service.build_user_profile_vector(
        user_interactions, data_map, prefs, tfidf_model
    )

    assert isinstance(vector, np.ndarray)
    assert np.all(vector == 0.0)
