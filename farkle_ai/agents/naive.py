"""Support for NaiveAgent."""

import numpy as np
from farkle_ai.game import to_dict_observation
from .agent import Agent


class NaiveAgent(Agent):
    """An agent with a simple naive heuristic."""

    def compute_action(self, observation: np.ndarray, mask: np.ndarray) -> int:
        """Choose highest scoring and continue rolling if remaining dice > 3."""

        observation_dict = to_dict_observation(observation)
        used_dice_count = (
            int(np.argmax(observation_dict["scores_by_pattern_length"])) + 1
        )
        remaining_dice_count = observation_dict["dice_count_in_roll"] - used_dice_count

        if remaining_dice_count > 3 or remaining_dice_count == 0:
            return used_dice_count - 1  # continue
        return used_dice_count - 1 + 6  # bank
