from datetime import datetime, timedelta, timezone

import pytest

from app.services.interaction_service import InteractionService


@pytest.fixture
def service():
    return InteractionService()


@pytest.fixture
def mock_create(mocker):
    return mocker.patch("app.services.interaction_service.create_document")


@pytest.fixture
def mock_list(mocker):
    return mocker.patch("app.services.interaction_service.list_documents")


@pytest.fixture
def mock_update(mocker):
    return mocker.patch("app.services.interaction_service.update_document")


def test_log_interaction_creates_new_doc(service, mock_create, mock_list):
    mock_list.return_value = {"documents": []}
    result = service.log_interaction("user123", "itemABC")
    mock_create.assert_called_once()
    assert result == "View logged."


def test_log_interaction_ignores_recent_duplicate(service, mock_list, mock_update):
    recent_time = datetime.now(timezone.utc).isoformat()
    mock_list.return_value = {
        "documents": [{"$id": "doc123", "timestamps": [recent_time]}]
    }
    result = service.log_interaction("user123", "itemABC")
    mock_update.assert_not_called()
    assert result == "duplicate_ignored"


def test_log_interaction_updates_old_view(service, mock_list, mock_update):
    old_time = (datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat()
    mock_list.return_value = {
        "documents": [{"$id": "doc123", "timestamps": [old_time]}]
    }
    result = service.log_interaction("user123", "itemABC")
    mock_update.assert_called_once()
    assert result == "View logged."


def test_log_interaction_prunes_old_timestamps(service, mock_list, mock_update):
    old_ts = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    recent_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    mock_list.return_value = {
        "documents": [{"$id": "doc456", "timestamps": [old_ts, recent_ts]}]
    }
    service.log_interaction("user123", "itemXYZ")
    updated_ts = mock_update.call_args[1]["data"]["timestamps"]
    threshold = datetime.now(timezone.utc) - timedelta(days=7)
    assert all(datetime.fromisoformat(ts) >= threshold for ts in updated_ts)
