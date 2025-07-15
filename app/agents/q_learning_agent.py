import pickle
import random
from collections import defaultdict
from typing import List


class QLearningAgent:
    def __init__(
        self,
        actions: List[str],
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
    ):
        self.actions = actions
        self.q_table = defaultdict(lambda: {a: 0.0 for a in actions})
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, state_key: str) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def update(self, state_key: str, action: str, reward: float, next_state_key: str):
        current_q = self.q_table[state_key][action]
        max_future_q = max(self.q_table[next_state_key].values(), default=0.0)
        new_q = current_q + self.alpha * (
            reward + self.gamma * max_future_q - current_q
        )
        self.q_table[state_key][action] = new_q

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filepath: str):
        try:
            with open(filepath, "rb") as f:
                table = pickle.load(f)
                self.q_table = defaultdict(
                    lambda: {a: 0.0 for a in self.actions}, table
                )
        except FileNotFoundError:
            pass
