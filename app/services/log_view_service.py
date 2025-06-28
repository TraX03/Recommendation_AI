from datetime import datetime, timedelta, timezone

from appwrite.query import Query

from app.constants import INTERACTIONS_COLLECTION_ID
from app.utils.appwrite_client import create_document, list_documents, update_document

VIEW_TYPE = "view"
COOLDOWN_MINUTES = 10
NOW = datetime.now(timezone.utc)
SEVEN_DAYS_AGO = NOW - timedelta(days=7)


def get_existing_view_interaction(user_id: str, item_id: str, interaction_type: str):
    queries = [
        Query.equal("user_id", user_id),
        Query.equal("item_id", item_id),
        Query.equal("type", interaction_type),
    ]
    documents = list_documents(
        collection_id=INTERACTIONS_COLLECTION_ID,
        queries=queries,
    )
    return documents[0] if documents else None


def log_user_view(user_id: str, item_id: str, source: str = "homeFeed") -> str:
    now = datetime.now(timezone.utc)

    existing_doc = get_existing_view_interaction(
        user_id=user_id, item_id=item_id, interaction_type=VIEW_TYPE
    )

    if not existing_doc:
        create_document(
            collection_id=INTERACTIONS_COLLECTION_ID,
            data={
                "user_id": user_id,
                "item_id": item_id,
                "type": VIEW_TYPE,
                "source": source,
                "created_at": now.isoformat(),
                "timestamps": [now.isoformat()],
            },
        )
        return "logged"

    timestamps = existing_doc.get("timestamps", [])
    if timestamps:
        last_view_time = datetime.fromisoformat(timestamps[-1])
        if now - last_view_time < timedelta(minutes=COOLDOWN_MINUTES):
            return "duplicate_ignored"

    updated_timestamps = timestamps + [now.isoformat()]

    update_document(
        document_id=existing_doc["$id"],
        collection_id=INTERACTIONS_COLLECTION_ID,
        data={"timestamps": updated_timestamps},
    )

    return "logged"
