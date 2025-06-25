import shelve
from typing import List

from app.utils.storage_utils import shelve_path


def save_last_recommendations(user_id: str, recipe_ids: List[str]) -> None:
    """Store a list of recommended recipe IDs for a user."""
    with shelve.open(shelve_path("last_recommendations")) as db:
        db[user_id] = [recipe_ids]


def get_last_recommendations(user_id: str) -> List[List[str]]:
    """Retrieve the list of recently recommended recipe IDs for a user."""
    with shelve.open(shelve_path("last_recommendations")) as db:
        return db.get(user_id, [])
