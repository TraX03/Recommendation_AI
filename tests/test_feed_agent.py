from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.agents import feed_agent
from app.agents.feed_agent import (
    calculate_feed_fallback_reward,
    choose_feed_fallback_action,
    encode_feed_state,
    update_feed_fallback_feedback,
)


@pytest.fixture
def prefs_base():
    return {
        "diet": [],
        "avoid_ingredients": [],
        "region_pref": ["asian"],
        "inferred_tags": [],
    }


@pytest.fixture
def interactions_low():
    return pd.DataFrame([{"id": i} for i in range(3)])


@pytest.fixture
def interactions_high():
    return pd.DataFrame([{"id": i} for i in range(10)])


def test_encode_feed_state_flexible(prefs_base, interactions_high):
    result = encode_feed_state(prefs_base, interactions_high)
    assert result == "feed_flexible_narrow_region_no_tags_high_activity"


def test_encode_feed_state_strict_diet(prefs_base, interactions_low):
    prefs = prefs_base.copy()
    prefs["diet"] = ["vegan"]
    result = encode_feed_state(prefs, interactions_low)
    assert result == "feed_strict_diet_narrow_region_no_tags_low_activity"


def test_encode_feed_state_has_avoids(prefs_base):
    prefs = prefs_base.copy()
    prefs["avoid_ingredients"] = ["milk"]
    result = encode_feed_state(prefs, pd.DataFrame())
    assert result.startswith("feed_has_avoids")


def test_encode_feed_state_broad_region_tags():
    prefs = {
        "diet": [],
        "avoid_ingredients": [],
        "region_pref": ["thai", "korean"],
        "inferred_tags": ["spicy"],
    }
    result = encode_feed_state(prefs, pd.DataFrame([{"id": 1}, {"id": 2}]))
    assert result == "feed_flexible_broad_region_has_tags_low_activity"


def test_choose_feed_fallback_action_calls_agent(
    monkeypatch, prefs_base, interactions_low
):
    mock_agent = MagicMock()
    mock_agent.get_action.return_value = "use_tags"
    monkeypatch.setattr(feed_agent, "feed_fallback_agent", mock_agent)
    result = choose_feed_fallback_action(prefs_base, interactions_low)
    assert result == "use_tags"
    assert mock_agent.get_action.called


@pytest.mark.parametrize(
    "interacted,liked,session,num_viewed,expected",
    [
        (True, True, 5, 5, 1.0),
        (True, False, 5, 5, 0.5),
        (False, False, 20, 10, -0.2),
        (False, False, 5, 2, -1.0),
    ],
)
def test_calculate_feed_fallback_reward(
    interacted, liked, session, num_viewed, expected
):
    reward = calculate_feed_fallback_reward(interacted, liked, session, num_viewed)
    assert reward == expected


def test_update_feed_fallback_feedback(monkeypatch, prefs_base, interactions_high):
    mock_agent = MagicMock()
    monkeypatch.setattr(feed_agent, "feed_fallback_agent", mock_agent)
    update_feed_fallback_feedback(
        prefs_base,
        interactions_high,
        action="use_tags",
        interacted=True,
        user_liked=True,
        session_duration=10,
        num_viewed_items=3,
    )
    assert mock_agent.update.called
    assert mock_agent.save.called
