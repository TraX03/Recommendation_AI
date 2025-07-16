from datetime import datetime, timezone

from app import dependencies


def start_session(user_id: str):
    dependencies.session_store[user_id] = datetime.now(timezone.utc)


def get_session_start(user_id: str):
    return dependencies.session_store.get(user_id)


def clear_session(user_id: str):
    dependencies.session_store.pop(user_id, None)
