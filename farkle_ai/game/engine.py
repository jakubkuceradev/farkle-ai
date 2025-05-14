"""Support for the Farkle game engine."""

import numpy as np
from .state import GameState
from .actions import Action, BankAction, ContinueAction, PassTurnAction
from .rules import (
    ScoringPattern,
    pattern_score,
    scorable_patterns,
    scorable_patterns_by_length,
    MAX_DICE_COUNT,
)


def actions(state: GameState) -> list[Action]:
    """Return permissible actions for a state."""

    possible_actions: list[Action] = []
    possible_actions.append(PassTurnAction())

    scoring_patterns = scorable_patterns(state.turn_state.rolled_dice)

    for scoring_pattern in scoring_patterns:
        possible_actions.append(ContinueAction(scoring_pattern.pattern))
        possible_actions.append(BankAction(scoring_pattern.pattern))

    return possible_actions


def observe(state: GameState, player: int) -> dict:
    """Create a summary for a given game state and player."""

    your_remaining_score = state.parameters.score_to_win - state.player_scores[player]
    opponent_remaining_score = state.parameters.score_to_win - max(
        score for index, score in enumerate(state.player_scores) if index != player
    )

    if state.current_player == player:
        turn_score = state.turn_state.score
        dice_count_in_roll = len(state.turn_state.rolled_dice)
        scores_by_pattern_length = [
            scoring_pattern.score
            for scoring_pattern in scorable_patterns_by_length(
                state.turn_state.rolled_dice
            )
        ]
    else:
        turn_score = 0
        dice_count_in_roll = 0
        scores_by_pattern_length = [0] * MAX_DICE_COUNT

    observation = {
        "your_remaining_score": your_remaining_score,
        "opponent_remaining_score": opponent_remaining_score,
        "turn_score": turn_score,
        "dice_count_in_roll": dice_count_in_roll,
        "scores_by_pattern_length": scores_by_pattern_length,
    }

    return observation

def to_dict_observation(observation: np.ndarray) -> dict:
    """Convert an ndarray observation into a dict representation."""

    observation_dict = {
        "your_remaining_score": observation[0],
        "opponent_remaining_score": observation[1],
        "turn_score": observation[2],
        "dice_count_in_roll": observation[3],
        "scores_by_pattern_length": observation[4:],
    }

    return observation_dict


def to_array_observation(observation: np.ndarray) -> dict:
    """Convert a dict observation into an ndarray representation."""

    observation = np.empty((4 + MAX_DICE_COUNT,), dtype=np.float32)
    observation[0] = max(0, observation["your_remaining_score"])
    observation[1] = max(0, observation["opponent_remaining_score"])
    observation[2] = observation["turn_score"]
    observation[3] = observation["dice_count_in_roll"]
    observation[4:] = observation["scores_by_pattern_length"]

    return observation


def apply_action(state: GameState, action: Action, rng=None) -> GameState:
    """Modify game state by applying an action."""

    if rng is None:
        rng = np.random.default_rng()

    new_state = state

    if isinstance(action, (ContinueAction, BankAction)):
        score = pattern_score(action.pattern)
        if score == 0:
            raise ValueError(
                f"Action pattern {action.pattern} must be a valid scoring pattern"
            )
        scoring_pattern = ScoringPattern(action.pattern, score)
        new_state = new_state.select_pattern(scoring_pattern)

    if isinstance(action, ContinueAction):
        new_state = new_state.roll_dice(rng)

    if isinstance(action, BankAction):
        new_state = new_state.end_turn()
        new_state = new_state.start_turn()
        new_state = new_state.roll_dice(rng)

    if isinstance(action, PassTurnAction):
        new_state = new_state.pass_turn()
        new_state = new_state.start_turn()
        new_state = new_state.roll_dice(rng)

    return new_state
