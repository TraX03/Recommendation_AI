import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

from app.constants import CONTENT_TYPE_MAP, USERS_COLLECTION_ID
from app.utils.appwrite_client import update_document

CACHE_DIR = "app/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "inferred_tags_cache.json")
MAX_AGE_DAYS = 4

os.makedirs(CACHE_DIR, exist_ok=True)


def get_inferred_tags(
    user_id: str,
    interactions_df: pd.DataFrame,
    data_map: dict,
    max_age_days: int = MAX_AGE_DAYS,
) -> List[str]:
    tag_cache = _load_tag_cache()
    now = datetime.now()

    cached = tag_cache.get(user_id)
    if cached and (now - cached["ts"]) < timedelta(days=max_age_days):
        return cached["tags"]

    user_interactions = interactions_df.query("user_id == @user_id")
    if user_interactions.empty:
        return []

    all_tags = []
    for ctype, df in data_map.items():
        id_col = CONTENT_TYPE_MAP[ctype]["id_col"]
        type_interactions = user_interactions[user_interactions["item_type"] == ctype]
        tags = _extract_tags(type_interactions, df, id_col)
        all_tags.extend(tags)

    unique_tags = list(set(all_tags))

    update_document(
        collection_id=USERS_COLLECTION_ID,
        document_id=user_id,
        data={"inferred_tags": unique_tags},
    )

    tag_cache[user_id] = {"tags": unique_tags, "ts": now}
    _save_tag_cache(tag_cache)

    return unique_tags


def _load_tag_cache() -> Dict[str, Dict]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            raw = json.load(f)
        return {
            uid: {"tags": data["tags"], "ts": datetime.fromisoformat(data["ts"])}
            for uid, data in raw.items()
        }
    return {}


def _save_tag_cache(cache: Dict[str, Dict]):
    serializable = {
        uid: {"tags": data["tags"], "ts": data["ts"].isoformat()}
        for uid, data in cache.items()
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(serializable, f)


def _extract_tags(
    interactions: pd.DataFrame, content_df: pd.DataFrame, id_col: str
) -> List[str]:
    if "tags" not in content_df.columns:
        return []
    matched = content_df[content_df[id_col].isin(interactions["item_id"].unique())]
    return (
        matched["tags"].explode().dropna().unique().tolist()
        if not matched.empty
        else []
    )
