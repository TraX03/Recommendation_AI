import json
import re

import pandas as pd
from fastapi import HTTPException

from app.constants import (
    COMMUNITIES_COLLECTION_ID,
    INTERACTIONS_COLLECTION_ID,
    POSTS_COLLECTION_ID,
    RECIPES_COLLECTION_ID,
    USERS_COLLECTION_ID,
)
from app.utils.appwrite_client import fetch_documents, get_document_by_id


def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


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
                "author_id": doc.get("author_id"),
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


def get_user_preferences(user_id: str) -> dict:
    try:
        user = get_document_by_id(USERS_COLLECTION_ID, user_id)
        return {
            "avoid_ingredients": user.get("avoid_ingredients", []),
            "diet": user.get("diet", []),
            "region_pref": user.get("region_pref", []),
        }
    except Exception:
        raise HTTPException(status_code=404, detail="User not found")
