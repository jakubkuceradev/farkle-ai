"""Support for AbstractAction."""

from enum import Enum
import numpy as np
from farkle_ai.game import GameState, scorable_patterns_by_length, MAX_DICE_COUNT


class AbstractAction(Enum):
    """An action space for abstract farkle actions."""

    # Actions 0-5: Continue holding 1 to 6 dice
    CONTINUE_1 = 0
    CONTINUE_2 = 1
    CONTINUE_3 = 2
    CONTINUE_4 = 3
    CONTINUE_5 = 4
    CONTINUE_6 = 5
    # Actions 6-11: Bank holding 1 to 6 dice
    BANK_1 = 6
    BANK_2 = 7
    BANK_3 = 8
    BANK_4 = 9
    BANK_5 = 10
    BANK_6 = 11

    @property
    def type(self) -> str:
        """Returns the action type."""
        return "continue" if self.value <= AbstractAction.CONTINUE_6.value else "bank"

    @property
    def dice_count(self) -> int:
        """Returns the number of dice this action corresponds to."""
        return (
            self.value + 1
            if self.type == "continue"
            else self.value - AbstractAction.BANK_1.value + 1
        )


def action_mask(state: GameState) -> np.ndarray:
    """Return an action mask for a given state and player."""

    mask = np.zeros(len(AbstractAction), dtype=np.int8)
    best_patterns_by_count = scorable_patterns_by_length(state.turn_state.rolled_dice)

    for i in range(MAX_DICE_COUNT):
        dice_count = i + 1
        scoring_pattern = best_patterns_by_count[i]

        is_valid_pattern_available = (
            (scoring_pattern is not None)
            and (scoring_pattern.score > 0)
            and (len(scoring_pattern.pattern) == dice_count)
        )

        if is_valid_pattern_available:
            continue_action_value = AbstractAction[f"CONTINUE_{dice_count}"].value
            bank_action_value = AbstractAction[f"BANK_{dice_count}"].value

            mask[continue_action_value] = True
            mask[bank_action_value] = True

    return mask
