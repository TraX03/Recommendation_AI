from typing import Any, Dict, List

import pandas as pd

from app.agents.feed_agent import update_feed_fallback_feedback
from app.agents.inventory_agent import update_inventory_feedback


class FeedbackService:
    def log_home_session_feedback(
        self,
        user_id: str,
        prefs: dict,
        session_start: pd.Timestamp,
        strategy: str,
        interactions_df: pd.DataFrame,
    ) -> str:
        now = pd.Timestamp.now(tz="UTC")
        session_duration = (now - session_start).total_seconds()

        recent = interactions_df[
            (interactions_df["user_id"] == user_id)
            & (pd.to_datetime(interactions_df["created_at"]) >= session_start)
        ]
        interacted = recent["type"].isin(["like", "bookmark", "view"]).any()
        user_liked = recent["type"].isin(["like", "bookmark"]).any()

        update_feed_fallback_feedback(
            prefs=prefs,
            interactions=recent,
            action=strategy,
            interacted=interacted,
            user_liked=user_liked,
            session_duration=session_duration,
            num_viewed_items=len(recent),
        )

        return "Feedback logged."

    def log_inventory_feedback(
        self,
        session_data: List[Dict[str, Any]],
        is_regenerate: bool = False,
    ) -> str:
        update_inventory_feedback(session_data, is_regenerate)
        return "Inventory feedback logged."
