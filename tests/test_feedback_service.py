from datetime import timedelta

import pandas as pd
import pytest

from app.services.feedback_service import FeedbackService


@pytest.fixture
def service():
    return FeedbackService()


@pytest.fixture
def mock_feed_agent(mocker):
    return mocker.patch("app.services.feedback_service.update_feed_fallback_feedback")


@pytest.fixture
def mock_inventory_agent(mocker):
    return mocker.patch("app.services.feedback_service.update_inventory_feedback")


def test_log_home_session_feedback_calls_update_correctly(service, mock_feed_agent):
    now = pd.Timestamp.now(tz="UTC")
    start = now - timedelta(minutes=5)
    df = pd.DataFrame(
        [
            {"user_id": "u1", "type": "view", "created_at": now.isoformat()},
            {"user_id": "u1", "type": "like", "created_at": now.isoformat()},
            {"user_id": "u2", "type": "view", "created_at": now.isoformat()},
        ]
    )
    prefs = {"cuisine": ["malay"]}
    result = service.log_home_session_feedback("u1", prefs, start, "content", df)
    mock_feed_agent.assert_called_once()
    assert result == "Feedback logged."


def test_log_home_session_feedback_no_match(service, mock_feed_agent):
    now = pd.Timestamp.now(tz="UTC")
    start = now - timedelta(minutes=5)
    df = pd.DataFrame(
        [
            {"user_id": "u2", "type": "view", "created_at": now.isoformat()},
        ]
    )
    prefs = {}
    result = service.log_home_session_feedback("u1", prefs, start, "tfidf", df)
    args = mock_feed_agent.call_args[1]

    assert not args["interacted"]
    assert not args["user_liked"]
    assert result == "Feedback logged."


def test_log_inventory_feedback_default(service, mock_inventory_agent):
    session_data = [{"item": "rice"}]
    result = service.log_inventory_feedback(session_data)
    mock_inventory_agent.assert_called_once_with(session_data, False)
    assert result == "Inventory feedback logged."


def test_log_inventory_feedback_with_regenerate(service, mock_inventory_agent):
    session_data = [{"item": "egg"}]
    result = service.log_inventory_feedback(session_data, is_regenerate=True)
    mock_inventory_agent.assert_called_once_with(session_data, True)
    assert result == "Inventory feedback logged."
