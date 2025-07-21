import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
from appwrite.query import Query
from fastapi import HTTPException

from app.constants import (
    COMMUNITIES_COLLECTION_ID,
    INTERACTIONS_COLLECTION_ID,
    LISTS_COLLECTION_ID,
    MEALPLAN_COLLECTION_ID,
    POSTS_COLLECTION_ID,
    RECIPES_COLLECTION_ID,
    USERS_COLLECTION_ID,
)
from app.utils.appwrite_client import (
    fetch_documents,
    get_document_by_id,
    list_documents,
    update_document,
)


def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_data_map() -> Dict[str, pd.DataFrame]:
    from app.utils.embedding_utils import (
        load_or_embed,
    )

    tips_df, discussion_df = fetch_post_data()

    return {
        "recipe": load_or_embed("recipes", fetch_recipe_data),
        "tip": load_or_embed("tips", lambda: tips_df),
        "discussion": load_or_embed("discussions", lambda: discussion_df),
        "community": load_or_embed("communities", fetch_community_data),
        "inventory": load_or_embed(
            "inventory", fetch_inventory_data, text_column="name"
        ),
        "interaction": fetch_interaction_data(),
    }


def fetch_recipe_data() -> pd.DataFrame:
    documents = fetch_documents(RECIPES_COLLECTION_ID)
    data = []

    for doc in documents:
        try:
            ingredients = [json.loads(i)["name"] for i in doc.get("ingredients", [])]
        except Exception:
            ingredients = []

        title = doc.get("title")
        tags = doc.get("tags", [])
        mealtime = doc.get("mealtime", [])
        category = doc.get("category", [])
        if not isinstance(category, list):
            category = [category]
        cuisine = doc.get("area", "")
        description = doc.get("description", "") or ""

        instructions = [
            step.get("text", "")
            for step in doc.get("instructions", [])
            if isinstance(step, dict)
        ]

        combined_text = " ".join(
            map(
                clean,
                [title or ""]
                + ingredients
                + tags
                + mealtime
                + category
                + [cuisine]
                + [description]
                + instructions,
            )
        )

        image = doc.get("image", [""])[0] if doc.get("image") else ""

        data.append(
            {
                "recipe_id": doc["$id"],
                "title": title,
                "category": category,
                "combined_text": combined_text,
                "image": image,
                "ingredients": ingredients,
                "author_id": doc.get("author_id"),
                "mealtime": mealtime,
                "tags": tags,
            }
        )

    return pd.DataFrame(data)


def fetch_post_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    documents = fetch_documents(POSTS_COLLECTION_ID)
    tips_data = []
    discussion_data = []

    for doc in documents:
        post_type = doc.get("type", "").lower()

        title = doc.get("title", "")
        content = doc.get("content", "")
        tags = doc.get("tags", [])

        post = {
            "post_id": doc["$id"],
            "title": title,
            "author_id": doc.get("author_id", ""),
            "image": doc.get("image", [""])[0] if doc.get("image") else "",
            "combined_text": " ".join(map(clean, [title, content] + tags)),
            "tags": tags,
        }

        if post_type == "tips":
            tips_data.append(post)
        elif post_type == "discussion":
            discussion_data.append(post)

    tips_df = pd.DataFrame(tips_data)
    discussion_df = pd.DataFrame(discussion_data)

    return tips_df, discussion_df


def fetch_community_data() -> pd.DataFrame:
    documents = fetch_documents(COMMUNITIES_COLLECTION_ID)
    data = []

    for doc in documents:
        name = doc.get("name", "")
        description = doc.get("description", "")
        tags = doc.get("tags", [])

        data.append(
            {
                "community_id": doc["$id"],
                "name": name,
                "image": doc.get("image", ""),
                "combined_text": " ".join(map(clean, [name, description] + tags)),
                "tags": tags,
            }
        )

    return pd.DataFrame(data)


def fetch_interaction_data() -> pd.DataFrame:
    documents = fetch_documents(INTERACTIONS_COLLECTION_ID)
    return pd.DataFrame(
        [
            {
                "interaction_id": d["$id"],
                "user_id": d.get("user_id"),
                "item_id": d.get("item_id"),
                "type": d.get("type"),
                "item_type": d.get("item_type"),
                "score": d.get("score"),
                "timestamps": d.get("timestamps"),
                "created_at": d.get("created_at"),
            }
            for d in documents
        ]
    )


def fetch_inventory_data() -> pd.DataFrame:
    documents = fetch_documents(LISTS_COLLECTION_ID)
    inventory_docs = [
        d for d in documents if d.get("type") == "inventory" and d.get("name")
    ]

    if not inventory_docs:
        return pd.DataFrame(
            columns=["user_id", "name", "unit", "quantity", "checkedCount", "expiries"]
        )

    return pd.DataFrame(
        [
            {
                "user_id": d.get("user_id"),
                "name": d.get("name"),
                "unit": d.get("unit"),
                "quantity": d.get("quantity"),
                "checkedCount": d.get("checkedCount", 0),
                "expiries": d.get("expiries"),
            }
            for d in inventory_docs
        ]
    )


def fetch_mealplan_data() -> pd.DataFrame:
    documents = fetch_documents(MEALPLAN_COLLECTION_ID)

    result = []
    for doc in documents:
        parsed_sessions = []
        session_entries = doc.get("session_data", [])
        for s in session_entries:
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list) and parsed:
                    parsed_sessions.append(parsed[0])
            except (json.JSONDecodeError, TypeError):
                continue

        result.append(
            {
                "$id": doc.get("$id"),
                "user_id": doc.get("user_id"),
                "session_data": parsed_sessions,
            }
        )

    return pd.DataFrame(result)


def fetch_session_data(
    user_id: Optional[str] = None,
    date: Optional[str] = None,
    mealtime: Optional[List[str]] = None,
    target_recipe_id: Optional[str] = None,
) -> Optional[List[dict]]:
    queries = []
    if user_id:
        queries.append(Query.equal("user_id", user_id))
    if date:
        queries.append(Query.equal("date", date))

    documents = fetch_documents(MEALPLAN_COLLECTION_ID, custom_queries=queries)
    if not documents:
        return None

    parsed: List[dict] = []

    for doc in documents:
        session_entries = doc.get("session_data", [])
        for s in session_entries:
            try:
                parsed_entry = json.loads(s)
                if isinstance(parsed_entry, list) and parsed_entry:
                    parsed.append(parsed_entry[0])
            except (json.JSONDecodeError, TypeError):
                continue

    if target_recipe_id:
        filtered = [entry for entry in parsed if entry.get("id") == target_recipe_id]
        return filtered or parsed

    if mealtime:
        return [entry for entry in parsed if entry.get("mealtime") in mealtime]

    return parsed


def get_user_preferences(user_id: str) -> dict:
    try:
        user = get_document_by_id(USERS_COLLECTION_ID, user_id)

        meal_config_raw = user.get("meal_config", "{}")
        if isinstance(meal_config_raw, str):
            try:
                meal_config = json.loads(meal_config_raw)
            except json.JSONDecodeError:
                meal_config = {}
        else:
            meal_config = meal_config_raw or {}

        return {
            "avoid_ingredients": user.get("avoid_ingredients", []),
            "diet": user.get("diet", []),
            "region_pref": user.get("region_pref", []),
            "meal_config": meal_config,
            "inferred_tags": user.get("inferred_tags", []),
        }

    except Exception:
        raise HTTPException(status_code=404, detail="User not found")


def get_mealplan_exclude_ids(user_id: str, target_date: date) -> List[str]:
    try:
        results = list_documents(
            MEALPLAN_COLLECTION_ID,
            [
                Query.equal("user_id", user_id),
                Query.equal("date", target_date.isoformat()),
                Query.limit(1),
            ],
        )

        documents = results.get("documents", [])
        if not documents:
            return []
        doc = documents[0]
        doc_id = doc["$id"]
        recommended_ids = doc.get("recommended_ids", [])
        recommended_ts = doc.get("recommended_ts")

        if recommended_ids and recommended_ts:
            try:
                ts = datetime.fromisoformat(recommended_ts)
                if datetime.now(timezone.utc) - ts > timedelta(minutes=20):
                    update_document(
                        MEALPLAN_COLLECTION_ID,
                        doc_id,
                        {
                            "recommended_ids": [],
                            "recommended_ts": None,
                        },
                    )
                    return []
            except Exception:
                update_document(
                    MEALPLAN_COLLECTION_ID,
                    doc_id,
                    {
                        "recommended_ids": [],
                        "recommended_ts": None,
                    },
                )
                return []

        return recommended_ids

    except Exception as e:
        print("[MealPlan] Error fetching meal plan:", repr(e))
        raise HTTPException(status_code=500, detail="Failed to fetch meal plan")
