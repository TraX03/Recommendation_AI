from typing import Any, Dict, List, Optional

import pandas as pd

from app.constants import CONTENT_TYPE_MAP
from app.models.schemas import PostList
from app.services.coldstart_service import ColdStartService
from app.services.collaborative_service import CollaborativeService
from app.services.content_based_service import ContentBasedService
from app.services.feedback_service import FeedbackService
from app.services.interaction_service import InteractionService
from app.utils.data_loader import load_data_map
from app.utils.session_utils import clear_session, get_session_start


class Recommender:
    def __init__(self):
        from app.services.hybrid_recommendation_service import (
            HybridRecommendationService,
        )

        self.coldstart_service = ColdStartService()
        self.interaction_service = InteractionService()
        self.content_service = ContentBasedService()
        self.collab_service = CollaborativeService()
        self.feedback_service = FeedbackService()

        self.hybrid_service = HybridRecommendationService(
            build_tfidf_model=self.content_service.build_tfidf_model,
            build_user_profile_vector=self.content_service.build_user_profile_vector,
        )

    def refresh_models(self):
        data_map = load_data_map()
        self.data_map = {
            k: data_map[k] for k in ["recipe", "tip", "discussion", "community"]
        }
        self.inventory_df = data_map["inventory"]
        self.interactions_df = data_map["interaction"]

        self.sim_models = self.content_service.prepare_cbf_models(self.data_map)
        self.cf_models = self.collab_service.prepare_cf_models(
            self.interactions_df, list(self.data_map.keys())
        )

    def cold_start(self, user_prefs: dict, max_recs: int = 10):
        recipe_df = self.coldstart_service.generate_coldstart(
            user_prefs=user_prefs,
            df=self.data_map["recipe"],
            id_column="recipe_id",
            avoid_filter=True,
            fit_new=True,
            max_recs=max_recs,
        )

        suggestions = {
            ctype: self.coldstart_service.generate_coldstart(
                user_prefs=user_prefs,
                df=self.data_map[ctype],
                id_column=config["id_col"],
                fit_new=False,
                max_recs=max_recs,
            )[config["id_col"]].tolist()
            for ctype, config in CONTENT_TYPE_MAP.items()
            if ctype != "recipe"
        }

        return {"recipe_df": recipe_df, "suggestions": suggestions}

    def log_interaction(
        self,
        user_id: str,
        item_id: str,
        item_type: str = "recipe",
        source: str = "homeFeed",
    ) -> str:
        return self.interaction_service.log_interaction(
            user_id=user_id,
            item_id=item_id,
            item_type=item_type,
            source=source,
        )

    def recommend(
        self,
        user_id: str,
        content_type: str,
        prefs: dict,
        max_count: int,
    ) -> PostList:
        interactions = self.interactions_df

        if "user_id" in interactions.columns:
            interactions = interactions[interactions["user_id"] == user_id].copy()
        else:
            interactions = pd.DataFrame()

        interactions["created_at"] = pd.to_datetime(
            interactions["created_at"], errors="coerce"
        )
        interactions = interactions.sort_values(by="created_at", ascending=False)

        return self.hybrid_service.generate_recommendations(
            user_id=user_id,
            content_type=content_type,
            prefs=prefs,
            interactions=interactions,
            data_map=self.data_map,
            sim_models=self.sim_models,
            cf_models=self.cf_models,
            max_count=max_count,
        )

    def generate_mealplan(
        self,
        user_id: str,
        mealtime: str,
        prefs: dict,
        exclude_ids: Optional[List[str]] = None,
        session_data: Optional[List[dict]] = None,
        target_recipe_id: Optional[str] = None,
    ) -> List[Dict]:
        return self.hybrid_service.generate_mealplan(
            mealtime=mealtime,
            config=prefs.get("meal_config", {}).get(mealtime, {}),
            avoid_ingredients=prefs.get("avoid_ingredients", []),
            region_pref=prefs.get("region_pref", []),
            diet=prefs.get("diet", []),
            recipe_df=self.data_map["recipe"],
            sim_model=self.sim_models["recipe"],
            cf_matrix=self.cf_models["recipe"],
            inventory_df=self.inventory_df.query("user_id == @user_id"),
            interactions_df=self.interactions_df.query("user_id == @user_id"),
            exclude_ids=exclude_ids or [],
            session_data=session_data,
            target_recipe_id=target_recipe_id,
        )

    def log_session_feedback(self, user_id: str, prefs: dict) -> str:
        session_start = get_session_start(user_id)
        if not session_start:
            print("[Session] No active session found for user:", user_id)
            return "No active session."

        strategy = self.hybrid_service.last_used_strategies.get(user_id)
        if not strategy:
            return "No strategy used."

        result = self.feedback_service.log_home_session_feedback(
            user_id=user_id,
            prefs=prefs,
            session_start=session_start,
            strategy=strategy,
            interactions_df=self.interactions_df,
        )

        clear_session(user_id)
        return result

    def log_inventory_feedback(
        self,
        session_data: List[Dict[str, Any]],
        is_regenerate: bool = False,
    ) -> str:
        if not session_data:
            return "No session data provided."

        result = self.feedback_service.log_inventory_feedback(
            session_data=session_data, is_regenerate=is_regenerate
        )

        return result
