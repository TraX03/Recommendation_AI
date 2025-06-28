import os
import shelve
from typing import List


def shelve_path(name: str) -> str:
    os.makedirs("app/data", exist_ok=True)
    return os.path.join("app/data", name)


def save_last_recommendations(user_id: str, recipe_ids: List[str]) -> None:
    with shelve.open(shelve_path("last_recommendations")) as db:
        db[user_id] = [recipe_ids]


def get_last_recommendations(user_id: str) -> List[List[str]]:
    with shelve.open(shelve_path("last_recommendations")) as db:
        return db.get(user_id, [])
