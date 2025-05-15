"""Support for agents and wrappers."""

from abc import ABC, abstractmethod
import numpy as np
from farkle_ai.game import (
    GameState,
    Action,
    PassTurnAction,
    ContinueAction,
    BankAction,
    scorable_patterns_by_length,
    to_array_observation,
    is_farkle,
    observe,
)
from farkle_ai.environment import AbstractAction, action_mask


class Agent(ABC):
    "An abstract farkle agent class."

    @abstractmethod
    def compute_action(self, observation: np.ndarray, mask: np.ndarray) -> int:
        """Given an observation and a valid action mask, compute the next action."""
        raise NotImplementedError

    def __str__(self):
        """
        A string representation, useful for printing results.
        Defaults to the class name.
        """
        return self.__class__.__name__


class AgentWrapper:
    """A wrapper to use abstract agents in a real game setting."""

    def __init__(self, agent: Agent):
        self.agent = agent

    def compute_action(self, state: GameState) -> Action:
        """Computes an Engine function using internal module."""

        if is_farkle(state.turn_state.rolled_dice):
            return PassTurnAction()

        observation_dict = observe(state, state.current_player)
        observation_array = to_array_observation(observation_dict)
        mask = action_mask(state)

        chosen_action = self.agent.compute_action(observation_array, mask)
        chosen_action_enum = AbstractAction(chosen_action)

        action_type = chosen_action_enum.type
        dice_count_to_hold = chosen_action_enum.dice_count

        by_length = scorable_patterns_by_length(state.turn_state.rolled_dice)
        scoring_pattern = by_length[dice_count_to_hold - 1]

        highest_scoring_dice_count = (
            int(np.argmax(observation_dict["scores_by_pattern_length"])) + 1
        )
        highest_scoring_pattern = by_length[highest_scoring_dice_count - 1]

        if action_type == "bank":
            engine_action: Action = BankAction(highest_scoring_pattern.pattern)
        else:  # continue
            engine_action = ContinueAction(scoring_pattern.pattern)

        return engine_action

    def __str__(self):
        return str(self.agent)
