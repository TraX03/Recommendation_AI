import json
import random
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale

from app.agents.feed_agent import choose_feed_fallback_action
from app.agents.inventory_agent import choose_inventory_action
from app.constants import (
    CONTENT_TYPE_MAP,
    RECOMMENDATION_DATA_COLLECTION_ID,
    TAG_CATEGORIES,
)
from app.models.schemas import PostList
from app.utils.appwrite_client import create_or_update_document, get_document_by_id
from app.utils.embedding_utils import embedding_model
from app.utils.filtering_utils import (
    filter_avoid_ingredients,
    filter_diet,
    filter_recent_seen,
    filter_region,
)
from app.utils.tag_utils import get_inferred_tags


class HybridRecommendationService:
    def __init__(self, build_user_profile_vector_fn: Callable):
        self.build_user_profile_vector = build_user_profile_vector_fn
        self.last_used_strategies = {}

    def get_last_strategy(self, user_id: str) -> Optional[str]:
        return self.last_used_strategies.get(user_id)

    def generate_recommendations(
        self,
        user_id: str,
        content_type: str,
        prefs: dict,
        interactions: pd.DataFrame,
        data_map: dict,
        sim_models: dict,
        cf_models: dict,
        max_count: int,
    ) -> PostList:
        id_col = CONTENT_TYPE_MAP[content_type]["id_col"]

        cf_matrix = cf_models.get(content_type)
        sim_model = sim_models.get(content_type)

        trending_ids = self._get_trending_ids(content_type, interactions, data_map)
        recent_ids = (
            interactions.query("item_type == @content_type")["item_id"]
            .dropna()
            .unique()
            .tolist()[:10]
        )
        seeds = recent_ids + [tid for tid in trending_ids if tid not in recent_ids][:2]

        threshold = 5 if content_type == "community" else 10
        if len(seeds) < threshold:
            seeds = self._coldstart_seeds(user_id, content_type, seeds)

        user_interactions = interactions.query("user_id == @user_id")

        recs = pd.concat(
            [
                self._score_items(
                    item_id=sid,
                    content_type=content_type,
                    prefs=prefs,
                    sim_model=sim_model,
                    cf_matrix=cf_matrix,
                    content_df=data_map[content_type],
                    user_interactions=user_interactions,
                )
                for sid in seeds
            ],
            ignore_index=True,
        ).drop_duplicates(subset=id_col)

        if content_type == "recipe":
            recs, region_filtered_out = self._apply_recipe_filters(recs, prefs)

        if len(recs) < max_count:
            prefs["inferred_tags"] = get_inferred_tags(
                user_id=user_id,
                interactions_df=interactions,
                data_map=data_map,
            )
            strategy = choose_feed_fallback_action(prefs, user_interactions)
            recs = self._apply_feed_fallback_strategy(
                strategy,
                recs,
                content_type,
                prefs,
                id_col,
                data_map,
                region_filtered_out,
            )

            if not hasattr(self, "last_used_strategies"):
                self.last_used_strategies = {}
            self.last_used_strategies[user_id] = strategy

        max_recent = 20 if content_type == "community" else 40
        recs = filter_recent_seen(
            recs, user_id, id_col, content_type, max_recent=max_recent
        )
        final = recs.head(max_count)

        create_or_update_document(
            RECOMMENDATION_DATA_COLLECTION_ID,
            user_id,
            {
                "user_id": user_id,
                f"last_recommendations_{content_type}": final[id_col].tolist(),
            },
        )

        return PostList(
            post_ids=final[id_col].tolist(),
            titles=final["title"].tolist(),
            images=final["image"].tolist(),
            author_ids=final["author_id"].tolist(),
        )

    def generate_mealplan(
        self,
        mealtime: str,
        config: Dict[str, Any],
        avoid_ingredients: List[str],
        region_pref: List[str],
        diet: List[str],
        recipe_df: pd.DataFrame,
        sim_model: dict,
        cf_matrix: dict,
        inventory_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        exclude_ids: Optional[List[str]] = None,
        session_data: Optional[List[dict]] = None,
        target_recipe_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        available_ingredients = inventory_df["name"].dropna().unique().tolist()
        near_expiry_cutoff = datetime.now() + timedelta(days=3)

        ingredient_expiry_map = {}
        for _, row in inventory_df.iterrows():
            name = row["name"]
            expiries = row.get("expiries", []) or []
            if isinstance(expiries, list) and expiries:
                try:
                    parsed = pd.to_datetime(expiries, errors="coerce").dropna()
                    if not parsed.empty:
                        ingredient_expiry_map[name] = parsed.min()
                except Exception:
                    continue

        valid_recipes = recipe_df[
            recipe_df["mealtime"].apply(
                lambda mt: self._resolve_mealtime_match(mt, mealtime)
            )
        ]
        if valid_recipes.empty:
            return {"recipes": [], "session_data": []}

        valid_ids = valid_recipes["recipe_id"].tolist()

        inventory_action = choose_inventory_action(mealtime, available_ingredients)

        recent_ids = (
            interactions_df.query("item_type == 'recipe'")["item_id"]
            .dropna()
            .unique()
            .tolist()
        )
        top_recent = [rid for rid in recent_ids if rid in valid_ids][:3]

        remaining_pool = [rid for rid in valid_ids if rid not in top_recent]
        sampled_valid = (
            pd.Series(remaining_pool).sample(min(2, len(remaining_pool))).tolist()
            if remaining_pool
            else []
        )

        combined = list(dict.fromkeys(top_recent + sampled_valid))
        if len(combined) < 5:
            additional = [rid for rid in valid_ids if rid not in combined][
                : 5 - len(combined)
            ]
            combined += additional

        top_seed_ids = combined
        all_recs = []
        for sid in top_seed_ids:
            scored = self._score_items(
                item_id=sid,
                content_type="recipe",
                prefs={"diet": diet, "region_pref": region_pref},
                sim_model=sim_model,
                cf_matrix=cf_matrix,
                content_df=recipe_df,
                user_interactions=interactions_df,
                item_cbf_weight=0.4,
                user_cbf_weight=0.3,
                cf_weight=0.3,
                top_k=100,
                sample_n=30,
            )
            scored = scored.merge(
                recipe_df[["recipe_id", "mealtime", "embedding", "ingredients"]],
                on="recipe_id",
                how="left",
            )
            all_recs.append(scored)

        if not all_recs:
            return {"recipes": [], "session_data": []}

        candidates = pd.concat(all_recs).drop_duplicates(subset="recipe_id")
        candidates = candidates[
            candidates["mealtime"].apply(
                lambda mt: self._resolve_mealtime_match(mt, mealtime)
            )
        ]
        candidates = filter_avoid_ingredients(candidates, avoid_ingredients)
        if exclude_ids:
            candidates = candidates[~candidates["recipe_id"].isin(exclude_ids)]

        candidates = self._apply_inventory_strategy(
            candidates=candidates,
            inventory_action=inventory_action,
            inventory_df=inventory_df,
            ingredient_expiry_map=ingredient_expiry_map,
            embedding_model=embedding_model,
            near_expiry_cutoff=near_expiry_cutoff,
        )

        now = datetime.now().isoformat()

        if target_recipe_id and session_data:
            selected, session = self._regenerate_one_recipe(
                candidates,
                session_data,
                target_recipe_id,
                config,
                mealtime,
                inventory_action,
                available_ingredients,
            )
            return {"recipes": selected, "session_data": session}

        selected = []
        is_combo_mealtime = mealtime in {"lunch", "dinner", "supper"}
        staple = config.get("staples", "none").lower()
        dish_count = config.get("dishCount", 1)

        if is_combo_mealtime:
            if staple != "none":
                selected += [
                    {**r, "source": "staple"}
                    for r in self._select_by_tags_or_category([staple], 1, candidates)
                ]
            else:
                dish_count = 0
        else:
            selected += [
                {**r, "source": "general"}
                for r in self._select_by_tags_or_category([], dish_count, candidates)
            ]

        for key in ["meat", "vege"]:
            count = config.get(f"{key}Count", 0)
            if count > 0:
                selected += [
                    {**r, "source": key}
                    for r in self._select_by_category_key(key, count, candidates)
                ]

        for key in ["soup", "side"]:
            if config.get(key):
                count = config.get(f"{key}Count", 1)
                selected += [
                    {**r, "source": key}
                    for r in self._select_by_category_key(key, count, candidates)
                ]

        session_data = [
            {
                "id": r["recipe_id"],
                "timestamp": now,
                "feedback": None,
                "mealtime": mealtime,
                "action": inventory_action,
                "available_ingredients": available_ingredients,
                "source": r["source"],
            }
            for r in selected
        ]

        return {
            "recipes": selected,
            "session_data": session_data,
        }

    def _regenerate_one_recipe(
        self,
        candidates: pd.DataFrame,
        session_data: List[Dict[str, Any]],
        target_recipe_id: str,
        config: Dict[str, Any],
        mealtime: str,
        inventory_action: str,
        available_ingredients: List[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        now = datetime.now().isoformat()
        matching = next((s for s in session_data if s["id"] == target_recipe_id), None)
        if not matching:
            return [], []

        source = matching.get("source")
        if not source:
            return [], []

        candidates = candidates[candidates["recipe_id"] != target_recipe_id]

        if source == "staple":
            new_dishes = self._select_by_tags_or_category(
                [config.get("staples", "none")], 1, candidates
            )
        else:
            new_dishes = self._select_by_category_key(source, 1, candidates)

        selected = [{**r, "source": source} for r in new_dishes]
        session = [
            {
                "id": r["recipe_id"],
                "timestamp": now,
                "feedback": None,
                "mealtime": mealtime,
                "action": inventory_action,
                "available_ingredients": available_ingredients,
                "source": source,
            }
            for r in selected
        ]
        return selected, session

    def _score_items(
        self,
        item_id: str,
        content_type: str,
        prefs: dict,
        sim_model: dict,
        cf_matrix: dict,
        content_df: pd.DataFrame,
        user_interactions: pd.DataFrame,
        item_cbf_weight=0.3,
        user_cbf_weight=0.4,
        cf_weight=0.3,
        top_k=100,
        sample_n=40,
    ) -> pd.DataFrame:
        if not sim_model or item_id not in sim_model["indices"]:
            return pd.DataFrame()

        # Item profile CBF scoring
        idx = sim_model["indices"][item_id]
        item_scores = minmax_scale(sim_model["cosine_sim"][idx])

        # CF scoring
        cf_scores = np.zeros_like(item_scores)
        if cf_matrix and item_id in cf_matrix["item_map"]:
            cf_idx = cf_matrix["item_map"][item_id]
            partial = cf_matrix["matrix"][cf_idx]
            for rid, partial_idx in cf_matrix["item_map"].items():
                full_idx = sim_model["indices"].get(rid)
                if full_idx is not None:
                    cf_scores[full_idx] = partial[partial_idx]
            cf_scores = minmax_scale(cf_scores)

        # User profile CBF scoring
        user_scores = minmax_scale(
            self.build_user_profile_vector(
                user_interactions=user_interactions,
                data_map={content_type: content_df},
                prefs=prefs,
                tfidf_model=sim_model,
            )
        )

        hybrid_scores = (
            item_cbf_weight * item_scores
            + user_cbf_weight * user_scores
            + cf_weight * cf_scores
        )

        top_indices = np.argsort(-hybrid_scores)[:top_k]
        selected = random.sample(list(top_indices), min(sample_n, len(top_indices)))

        df = content_df.copy()
        if content_type == "community" and "name" in df and "title" not in df:
            df["title"] = df["name"]

        for col in ["title", "image", "author_id"]:
            if col not in df.columns:
                df[col] = ""

        return df.iloc[selected][
            [
                CONTENT_TYPE_MAP[content_type]["id_col"],
                "title",
                "image",
                "author_id",
                "combined_text",
            ]
        ]

    def _get_trending_ids(
        self, content_type: str, interactions: pd.DataFrame, data_map: dict
    ) -> list:
        id_col = CONTENT_TYPE_MAP[content_type]["id_col"]
        top_ids = (
            interactions.query("type == 'like' and item_type == @content_type")
            .groupby("item_id")
            .size()
            .nlargest(10)
            .index
        )
        return data_map[content_type][data_map[content_type][id_col].isin(top_ids)][
            id_col
        ].tolist()

    def _coldstart_seeds(self, user_id: str, content_type: str, seeds: list) -> list:
        key = {"tip": "tip", "discussion": "discussion", "community": "community"}.get(
            content_type
        )
        try:
            doc = get_document_by_id(RECOMMENDATION_DATA_COLLECTION_ID, user_id)
            onboarding = json.loads(doc.get("onboarding_suggestions", "{}"))
        except Exception:
            onboarding = {}

        ids = onboarding.get(key, []) if key else []
        threshold = 5 if content_type == "community" else 10
        top_up = [i for i in ids if i not in seeds][: threshold - len(seeds)]
        return seeds + top_up

    def _apply_recipe_filters(
        self, recs: pd.DataFrame, prefs: dict, skip_region: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        filtered_out = pd.DataFrame()

        recs = filter_avoid_ingredients(recs, prefs.get("avoid_ingredients", []))
        recs = filter_diet(recs, prefs.get("diet"))

        if not skip_region:
            region_pref = prefs.get("region_pref", [])
            if region_pref:
                filtered_recs = filter_region(recs, region_pref)
                filtered_out = recs[~recs.index.isin(filtered_recs.index)]
                recs = filtered_recs

        return recs, filtered_out

    def _resolve_mealtime_match(self, mt_list: List[str], mealtime: str) -> bool:
        return mealtime in mt_list or "all" in mt_list

    def _select_by_tags_or_category(
        self, keywords: List[str], count: int, candidates: pd.DataFrame
    ) -> List[Dict]:
        if not keywords:
            if "score" in candidates.columns:
                sorted_candidates = candidates.sort_values(by="score", ascending=False)
            else:
                sorted_candidates = candidates.sample(frac=1)
            return sorted_candidates.head(count).to_dict("records")

        query = " ".join(keywords)
        query_embedding = embedding_model.encode(query)

        filtered = candidates[candidates["embedding"].notnull()].copy()
        if filtered.empty:
            return []

        candidate_embeddings = np.stack(filtered["embedding"].values)
        similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        filtered["similarity"] = similarities

        if "score" not in filtered.columns:
            filtered["score"] = 0.0
        else:
            filtered["score"] = filtered["score"].fillna(0.0)

        filtered["final_score"] = 0.6 * filtered["similarity"] + 0.4 * filtered["score"]

        top_matches = filtered.sort_values(by="final_score", ascending=False).head(
            count
        )
        return top_matches.to_dict("records")

    def _select_by_category_key(
        self, key: str, count: int, candidates: pd.DataFrame
    ) -> List[Dict]:
        keywords = TAG_CATEGORIES.get(key, [])
        if key != "side":
            keywords = [kw for kw in keywords if kw.lower() != "sauce"]
        return self._select_by_tags_or_category(keywords, count, candidates)

    def _apply_feed_fallback_strategy(
        self,
        strategy: str,
        recs: pd.DataFrame,
        content_type: str,
        prefs: dict,
        id_col: str,
        data_map: dict,
        region_filtered_out: pd.DataFrame = None,
    ) -> pd.DataFrame:
        skip_region = False

        if strategy == "relax_region" and content_type == "recipe":
            if region_filtered_out is not None:
                recs = pd.concat(
                    [recs, region_filtered_out], ignore_index=True
                ).drop_duplicates(subset=id_col)
            skip_region = True

        elif strategy == "use_tags":
            tags = prefs.get("inferred_tags", [])

            if "tags" in data_map[content_type].columns:
                data_map[content_type]["tags"] = data_map[content_type]["tags"].apply(
                    lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                )

                tag_matches = data_map[content_type][
                    data_map[content_type]["tags"].apply(
                        lambda tags_list: isinstance(tags_list, list)
                        and any(tag in tags_list for tag in tags)
                    )
                ]
                tag_matches = tag_matches.sample(frac=1, random_state=None)

                recs = pd.concat([recs, tag_matches]).drop_duplicates(subset=id_col)
            else:
                print("[WARN] 'tags' column missing in data")

        elif strategy == "use_full_dataset":
            full_df = data_map.get("recipe", pd.DataFrame())
            full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
            recs = pd.concat([recs, full_df]).drop_duplicates(subset=id_col)

        if content_type == "recipe":
            recs, _ = self._apply_recipe_filters(recs, prefs, skip_region=skip_region)

        return recs

    def _apply_inventory_strategy(
        self,
        candidates: pd.DataFrame,
        inventory_action: str,
        inventory_df: pd.DataFrame,
        ingredient_expiry_map: Dict[str, datetime],
        embedding_model,
        near_expiry_cutoff: datetime,
    ) -> pd.DataFrame:
        if inventory_action == "boost_inventory_match":
            inventory_embeddings = {
                row["name"]: np.array(row["embedding"])
                for _, row in inventory_df.iterrows()
                if isinstance(row.get("embedding"), (list, np.ndarray))
            }

            def semantic_inventory_match_score(ingredients):
                if (
                    not isinstance(ingredients, (list, np.ndarray))
                    or len(ingredients) == 0
                ):
                    return 0.0
                ingredient_vectors = embedding_model.encode(
                    ingredients, convert_to_numpy=True
                )
                match_scores = []
                for vec in ingredient_vectors:
                    similarities = [
                        cosine_similarity([vec], [inv_vec])[0][0]
                        for inv_vec in inventory_embeddings.values()
                    ]
                    if similarities:
                        match_scores.append(max(similarities))
                return sum(match_scores) / len(match_scores) if match_scores else 0.0

            candidates["score"] = 0.0
            candidates["semantic_score"] = candidates["ingredients"].apply(
                semantic_inventory_match_score
            )
            candidates["score"] += 0.7 * candidates["semantic_score"]

        elif inventory_action == "prioritize_near_expiry":

            def near_expiry_score(ingredients):
                for i in ingredients:
                    if (
                        i in ingredient_expiry_map
                        and ingredient_expiry_map[i] <= near_expiry_cutoff
                    ):
                        return 1.0
                return 0.0

            candidates["score"] = 0.0
            candidates["near_expiry_score"] = candidates["ingredients"].apply(
                near_expiry_score
            )
            candidates["score"] += candidates["near_expiry_score"]

        return candidates
