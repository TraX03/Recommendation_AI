from typing import Dict

import pandas as pd

from app.agents.q_learning_agent import QLearningAgent

STRICT_DIETS = {"vegetarian", "vegan", "halal", "kosher", "pescatarian"}

FALLBACK_ACTIONS = [
    "use_tags",
    "relax_region",
    "use_full_dataset",
]

feed_fallback_agent = QLearningAgent(actions=FALLBACK_ACTIONS)
feed_fallback_agent.load("app/cache/feed_fallback_q_table.pkl")


def encode_feed_state(prefs: Dict, interactions: pd.DataFrame) -> str:
    diet = prefs.get("diet", [])
    diet = [diet] if isinstance(diet, str) else diet

    strictness = (
        "strict_diet"
        if any(d in STRICT_DIETS for d in diet)
        else ("has_avoids" if prefs.get("avoid_ingredients") else "flexible")
    )

    region_scope = (
        "narrow_region" if len(prefs.get("region_pref", [])) <= 1 else "broad_region"
    )
    tag_flag = "has_tags" if prefs.get("inferred_tags") else "no_tags"
    session_activity = "low_activity" if len(interactions) < 5 else "high_activity"

    return f"feed_{strictness}_{region_scope}_{tag_flag}_{session_activity}"


def choose_feed_fallback_action(prefs: Dict, interactions: pd.DataFrame) -> str:
    state_key = encode_feed_state(prefs, interactions)
    return feed_fallback_agent.get_action(state_key)


def update_feed_fallback_feedback(
    prefs: Dict,
    interactions: pd.DataFrame,
    action: str,
    interacted: bool,
    user_liked: bool,
    session_duration: float = None,
    num_viewed_items: int = None,
):
    state_key = encode_feed_state(prefs, interactions)
    reward = calculate_feed_fallback_reward(
        interacted, user_liked, session_duration, num_viewed_items
    )
    feed_fallback_agent.update(state_key, action, reward, next_state_key=state_key)
    feed_fallback_agent.save("app/cache/feed_fallback_q_table.pkl")


def calculate_feed_fallback_reward(
    interacted: bool,
    user_liked: bool,
    session_duration: float = None,
    num_viewed_items: int = None,
) -> float:
    if interacted and user_liked:
        return 1.0
    elif interacted:
        return 0.5
    elif (session_duration or 0) > 15 and (num_viewed_items or 0) > 5:
        return -0.2
    else:
        return -1.0
