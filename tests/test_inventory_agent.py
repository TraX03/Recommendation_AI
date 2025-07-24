from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from app.agents import inventory_agent
from app.agents.inventory_agent import (
    calculate_inventory_reward,
    choose_inventory_action,
    encode_inventory_state,
    update_inventory_feedback,
)


@pytest.fixture
def mock_agent(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(inventory_agent, "inventory_agent", mock)
    return mock


def test_encode_inventory_state_basic():
    result = encode_inventory_state("lunch", ["rice", "carrot", "tofu"])
    assert result == "lunch_has_staple_low"


def test_encode_inventory_state_high_diversity():
    ingredients = [f"item{i}" for i in range(10)]
    result = encode_inventory_state("dinner", ingredients)
    assert result == "dinner_no_staple_high"


def test_encode_inventory_state_unknown():
    result = encode_inventory_state(None, [])
    assert result == "unknown_no_staple_low"


def test_choose_inventory_action_calls_agent(mock_agent):
    mock_agent.get_action.return_value = "boost_inventory_match"
    result = choose_inventory_action("breakfast", ["noodles", "egg"])
    assert result == "boost_inventory_match"
    assert mock_agent.get_action.called


def test_calculate_inventory_reward_regenerated():
    ts = datetime.now(timezone.utc)
    reward = calculate_inventory_reward("soft_reward", ts, regenerated=True)
    assert reward == -1.0


def test_calculate_inventory_reward_idle():
    ts = datetime.now(timezone.utc) - timedelta(hours=7)
    reward = calculate_inventory_reward(None, ts)
    assert reward == 0.5


def test_calculate_inventory_reward_default():
    ts = datetime.now(timezone.utc)
    reward = calculate_inventory_reward("soft_reward", ts)
    assert reward == 0.0


def test_update_inventory_feedback_applies_rewards(mock_agent):
    timestamp = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    session_data = [
        {
            "timestamp": timestamp,
            "feedback": None,
            "action": "boost_inventory_match",
            "available_ingredients": ["rice"],
            "mealtime": "lunch",
        }
    ]
    updated = update_inventory_feedback(session_data)
    assert len(updated) == 1
    assert updated[0]["feedback"] == "soft_reward"
    assert mock_agent.update.called
    assert mock_agent.save.called


def test_update_inventory_feedback_skips_invalid(mock_agent):
    session_data = [
        {"timestamp": None, "action": "boost_inventory_match"},
        {"timestamp": "invalid", "action": "boost_inventory_match"},
        {"timestamp": datetime.now(timezone.utc).isoformat(), "action": None},
    ]
    updated = update_inventory_feedback(session_data)
    assert updated == []
