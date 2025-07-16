import asyncio
import json
from datetime import datetime

from app.agents.inventory_agent import update_inventory_feedback
from app.constants import MEALPLAN_COLLECTION_ID
from app.utils.appwrite_client import update_document
from app.utils.data_loader import (
    fetch_mealplan_data,
)


async def apply_soft_feedback():
    print("[Feedback Sync] Looking for old unreviewed inventory sessions...")

    df = fetch_mealplan_data()
    now = datetime.now()

    for _, row in df.iterrows():
        document_id = row["$id"]
        user_id = row.get("user_id")
        session_entries = row["session_data"]
        if not session_entries:
            continue

        parsed_entries = []
        for s in session_entries:
            if isinstance(s, dict):
                parsed_entries.append(s)
            else:
                try:
                    parsed_entry = json.loads(s)
                    if isinstance(parsed_entry, list) and parsed_entry:
                        parsed_entries.append(parsed_entry[0])
                except (json.JSONDecodeError, TypeError):
                    continue

        stale_entries = [
            entry
            for entry in parsed_entries
            if entry.get("feedback") is None
            and "timestamp" in entry
            and (now - datetime.fromisoformat(entry["timestamp"])).total_seconds()
            > 6 * 3600
        ]

        if not stale_entries:
            continue

        print(
            f"[Feedback Sync] Applying soft reward to {len(stale_entries)} items for user {user_id}"
        )
        updated_entries = update_inventory_feedback(
            session_data=stale_entries, is_regenerate=False
        )

        if not updated_entries:
            continue

        updated_session_data = []
        for entry in parsed_entries:
            for updated in updated_entries:
                if entry.get("id") == updated.get("id"):
                    entry = updated
                    break
            updated_session_data.append(json.dumps([entry]))

        update_document(
            collection_id=MEALPLAN_COLLECTION_ID,
            document_id=document_id,
            data={"session_data": updated_session_data},
        )

    print("[Feedback Sync] Done.")


async def apply_soft_feedback_loop():
    while True:
        await apply_soft_feedback()
        await asyncio.sleep(60 * 60)


async def start_background_tasks():
    asyncio.create_task(apply_soft_feedback_loop())
