"""Support for RandomAgent."""

import numpy as np
from .agent import Agent


class RandomAgent(Agent):
    """An agent that chooses a valid action at random."""

    def compute_action(self, observation: np.ndarray, mask: np.ndarray) -> int:
        """Selects a random action from the valid actions indicated by the action mask."""

        valid_action_indices = np.where(mask == 1)[0]
        chosen_action_index = np.random.choice(valid_action_indices)
        return int(chosen_action_index)
