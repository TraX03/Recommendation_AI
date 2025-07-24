from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from dateutil.parser import isoparse

from app.agents.q_learning_agent import QLearningAgent

INVENTORY_ACTIONS = [
    "boost_inventory_match",
    "prioritize_near_expiry",
]

inventory_agent = QLearningAgent(actions=INVENTORY_ACTIONS)
inventory_agent.load("app/cache/inventory_q_table.pkl")


def encode_inventory_state(
    mealtime: Optional[str],
    available_ingredients: List[str],
) -> str:
    staple_keywords = {"rice", "noodles", "sandwich", "pasta", "burger"}
    has_staple = any(i for i in available_ingredients if i in staple_keywords)
    staple_token = "has_staple" if has_staple else "no_staple"
    ingredient_diversity = "high" if len(set(available_ingredients)) >= 10 else "low"

    return f"{mealtime or 'unknown'}_{staple_token}_{ingredient_diversity}"


def choose_inventory_action(
    mealtime: Optional[str],
    available_ingredients: List[str],
) -> str:
    state = encode_inventory_state(mealtime, available_ingredients)
    return inventory_agent.get_action(state)


def update_inventory_feedback(
    session_data: List[Dict[str, Any]],
    is_regenerate: bool = False,
) -> List[Dict[str, Any]]:
    updated = []

    for entry in session_data:
        timestamp_str = entry.get("timestamp")
        feedback = entry.get("feedback")
        action = entry.get("action")
        available_ingredients = entry.get("available_ingredients", [])
        mealtime = entry.get("mealtime")

        if not action or not timestamp_str:
            continue

        try:
            timestamp = isoparse(timestamp_str).astimezone(timezone.utc)
        except (ValueError, TypeError):
            continue

        if feedback is None:
            feedback = "soft_reward"
            entry["feedback"] = feedback
            updated.append(entry)

        state = encode_inventory_state(mealtime, available_ingredients)
        reward = calculate_inventory_reward(feedback, timestamp, is_regenerate)
        inventory_agent.update(state, action, reward, next_state_key=state)

    inventory_agent.save("app/cache/inventory_q_table.pkl")
    return updated


def calculate_inventory_reward(
    feedback: str,
    timestamp: datetime,
    regenerated: bool = False,
    idle_threshold: timedelta = timedelta(hours=6),
) -> float:
    now = datetime.now(timestamp.tzinfo or timezone.utc)

    if regenerated:
        return -1.0
    elif feedback is None and (now - timestamp) > idle_threshold:
        return 0.5
    else:
        return 0.0
