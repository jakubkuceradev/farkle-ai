"""Support for BankHighestAgent."""

import numpy as np
from .agent import Agent


class BankHighestAgent(Agent):
    """An agent that chooses the highest scoring bank action."""

    def compute_action(self, observation: np.ndarray, mask: np.ndarray) -> int:
        """Return a bank action with the highest score."""

        dice_count = int(np.argmax(observation[4:])) + 1
        action = dice_count - 1 + 6  # bank
        return action
