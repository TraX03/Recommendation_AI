from unittest.mock import patch

import pandas as pd
import pytest

from app.services.coldstart_service import ColdStartService


@pytest.fixture
def mock_dataframe():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "combined_text": "spicy indian vegetarian curry",
                "category": ["main"],
            },
            {
                "id": 2,
                "combined_text": "creamy italian pasta with cheese",
                "category": ["main"],
            },
            {
                "id": 3,
                "combined_text": "traditional malay chicken rendang",
                "category": ["sauce", "main"],
            },
            {
                "id": 4,
                "combined_text": "simple salad with tomato and basil",
                "category": ["starter"],
            },
            {
                "id": 5,
                "combined_text": "japanese sushi with rice and fish",
                "category": ["main"],
            },
        ]
    )


@pytest.fixture
def user_prefs():
    return {
        "diet": ["vegetarian"],
        "region_pref": ["indian"],
        "avoid_ingredients": ["cheese"],
    }


def test_generate_coldstart_basic(mock_dataframe, user_prefs):
    service = ColdStartService()
    service.tfidf.fit(mock_dataframe["combined_text"])

    recs = service.generate_coldstart(
        user_prefs=user_prefs,
        df=mock_dataframe,
        id_column="id",
        avoid_filter=False,
        fit_new=False,
        max_recs=3,
    )

    assert isinstance(recs, pd.DataFrame)
    assert len(recs) <= 3
    assert "similarity" in recs.columns


def test_generate_coldstart_with_avoid_filter(mock_dataframe, user_prefs):
    service = ColdStartService()
    service.tfidf.fit(mock_dataframe["combined_text"])

    filtered = service.generate_coldstart(
        user_prefs=user_prefs,
        df=mock_dataframe,
        id_column="id",
        avoid_filter=True,
        fit_new=False,
        max_recs=10,
    )

    assert not any(filtered["combined_text"].str.contains("cheese"))
    assert all(
        "sauce" not in " ".join(cat).lower()
        if isinstance(cat, list)
        else "sauce" not in str(cat).lower()
        for cat in filtered["category"]
    )


def test_generate_coldstart_with_fit_new(mock_dataframe, user_prefs):
    service = ColdStartService()

    recs = service.generate_coldstart(
        user_prefs=user_prefs,
        df=mock_dataframe,
        id_column="id",
        avoid_filter=False,
        fit_new=True,
        max_recs=2,
    )

    assert isinstance(recs, pd.DataFrame)
    assert len(recs) <= 2


def test_preferred_filter_logic(mock_dataframe):
    service = ColdStartService()
    diet = ["vegetarian"]
    region = ["indian"]

    assert (
        service._is_preferred_text("Spicy Indian vegetarian dish", diet, region) is True
    )

    assert (
        service._is_preferred_text("Spicy Indian chicken dish", diet, region) is False
    )

    assert service._is_preferred_text("Creamy vegetarian pasta", diet, region) is False

    assert service._is_preferred_text("Fish and chips", diet, region) is False


@patch("app.utils.filtering_utils.filter_avoid_ingredients")
def test_recipe_avoid_filter_excludes_sauces(mock_filter, mock_dataframe):
    service = ColdStartService()

    mock_filter.return_value = mock_dataframe
    filtered = service._recipe_avoid_filter(mock_dataframe, {})

    assert all(
        "sauce" not in " ".join(cat).lower()
        if isinstance(cat, list)
        else "sauce" not in str(cat).lower()
        for cat in filtered["category"]
    )
