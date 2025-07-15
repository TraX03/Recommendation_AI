from datetime import datetime, timedelta, timezone

from appwrite.query import Query

from app.constants import INTERACTIONS_COLLECTION_ID
from app.utils.appwrite_client import create_document, list_documents, update_document


class InteractionService:
    VIEW_TYPE = "view"

    def log_interaction(
        self,
        user_id: str,
        item_id: str,
        source: str = "homeFeed",
        item_type: str = "recipe",
    ) -> str:
        now = datetime.now(timezone.utc)

        existing_doc = self._get_existing_view_interaction(
            user_id=user_id, item_id=item_id, interaction_type=self.VIEW_TYPE
        )

        if not existing_doc:
            create_document(
                collection_id=INTERACTIONS_COLLECTION_ID,
                data={
                    "user_id": user_id,
                    "item_id": item_id,
                    "type": self.VIEW_TYPE,
                    "item_type": item_type,
                    "source": source,
                    "created_at": now.isoformat(),
                    "timestamps": [now.isoformat()],
                },
            )
            return "View logged."

        timestamps = existing_doc.get("timestamps", [])
        if timestamps:
            last_view_time = datetime.fromisoformat(timestamps[-1])
            if now - last_view_time < timedelta(minutes=10):
                return "duplicate_ignored"

        # Only keep timestamps within the last 7 days
        seven_days_ago = now - timedelta(days=7)
        updated_timestamps = [
            ts for ts in timestamps if datetime.fromisoformat(ts) >= seven_days_ago
        ]
        updated_timestamps.append(now.isoformat())

        update_document(
            document_id=existing_doc["$id"],
            collection_id=INTERACTIONS_COLLECTION_ID,
            data={"timestamps": updated_timestamps},
        )

        return "View logged."

    def _get_existing_view_interaction(
        self, user_id: str, item_id: str, interaction_type: str
    ):
        queries = [
            Query.equal("user_id", user_id),
            Query.equal("item_id", item_id),
            Query.equal("type", interaction_type),
        ]
        documents = list_documents(
            collection_id=INTERACTIONS_COLLECTION_ID,
            queries=queries,
        )

        return documents["documents"][0] if documents["documents"] else None
