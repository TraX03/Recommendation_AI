import json
import os
from typing import Tuple

import pandas as pd

from app.utils.appwrite_client import fetch_documents


def fetch_recipe_data() -> pd.DataFrame:
    documents = fetch_documents(os.environ["APPWRITE_RECIPES_COLLECTION_ID"])
    data = []

    for doc in documents:
        try:
            ingredients = [json.loads(i)["name"] for i in doc.get("ingredients", [])]
        except Exception:
            ingredients = []

        tags = doc.get("tags", [])
        mealtime = doc.get("mealtime", [])
        cuisine = doc.get("area", "") or doc.get("category", "")
        description = doc.get("description", "") or ""

        instructions = [
            step.get("text", "")
            for step in doc.get("instructions", [])
            if isinstance(step, dict)
        ]

        combined_text = " ".join(
            ingredients + tags + mealtime + [cuisine, description] + instructions
        )

        image = doc.get("image", [""])[0] if doc.get("image") else ""

        data.append(
            {
                "recipe_id": doc["$id"],
                "title": doc.get("title"),
                "combined_text": combined_text,
                "image": image,
                "author_id": doc.get("author_id"),
            }
        )

    return pd.DataFrame(data)


def fetch_interaction_data() -> Tuple[pd.DataFrame, bool]:
    documents = fetch_documents(os.environ["APPWRITE_INTERACTIONS_COLLECTION_ID"])
    df = pd.DataFrame(
        [
            {
                "interaction_id": d["$id"],
                "user_id": d.get("user_id"),
                "recipe_id": d.get("item_id"),
                "type": d.get("type"),
                "value": d.get("value"),
                "created_at": d.get("created_at"),
            }
            for d in documents
        ]
    )

    used_mock = False
    if len(df) < 1000:
        mock_path = "mockData/user_interactions.xlsx"
        if os.path.exists(mock_path):
            mock_df = pd.read_excel(mock_path)
            df = pd.concat([df, mock_df], ignore_index=True)
            used_mock = True

    return df, used_mock
