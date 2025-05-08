"""Support for the Farkle game engine."""

from dataclasses import dataclass, field
from .state import GameState
from .actions import Action, BankAction, ContinueAction, PassTurnAction
from .rules import (
    ScoringPattern,
    scoring_patterns_table,
    scoring_patterns_for_roll,
)


@dataclass(frozen=True)
class FarkleEngine:
    """An engine for the farkle dice game."""

    pattern_scores: dict[tuple[int, ...], int] = field(
        default_factory=scoring_patterns_table
    )

    def actions(self, state: GameState) -> list[Action]:
        """Return permissible actions for a state."""

        possible_actions: list[Action] = []
        possible_actions.append(PassTurnAction())

        scoring_patterns = scoring_patterns_for_roll(
            state.turn_state.rolled_dice, self.pattern_scores
        )

        for scoring_pattern in scoring_patterns:
            possible_actions.append(ContinueAction(scoring_pattern.pattern))
            possible_actions.append(BankAction(scoring_pattern.pattern))

        return possible_actions

    def score_pattern(self, pattern: tuple[int, ...]) -> ScoringPattern:
        """Returns the ScoringPattern for a given pattern."""
        return ScoringPattern(
            pattern=pattern,
            score=self.pattern_scores.get(pattern, 0),
        )

    def result(self, state: GameState, action: Action) -> GameState:
        """Peek at the resulting game state for a given action and state."""

        new_state = state

        if isinstance(action, (ContinueAction, BankAction)):
            score = self.pattern_scores.get(action.pattern, None)
            if score is None:
                raise ValueError(
                    f"Action pattern {action.pattern} must be a valid scoring pattern"
                )
            scoring_pattern = ScoringPattern(action.pattern, score)
            new_state = new_state.select_pattern(scoring_pattern)

        if isinstance(action, BankAction):
            new_state = new_state.end_turn()

        if isinstance(action, PassTurnAction):
            new_state = new_state.pass_turn()

        return new_state

    def apply_action(self, state: GameState, action: Action) -> GameState:
        """Modify game state by applying an action."""

        new_state = state

        if isinstance(action, (ContinueAction, BankAction)):
            score = self.pattern_scores.get(action.pattern, None)
            if score is None:
                raise ValueError(
                    f"Action pattern {action.pattern} must be a valid scoring pattern"
                )
            scoring_pattern = ScoringPattern(action.pattern, score)
            new_state = new_state.select_pattern(scoring_pattern)

        if isinstance(action, ContinueAction):
            new_state = new_state.roll_dice()

        if isinstance(action, BankAction):
            new_state = new_state.end_turn()
            new_state = new_state.start_turn()
            new_state = new_state.roll_dice()

        if isinstance(action, PassTurnAction):
            new_state = new_state.pass_turn()
            new_state = new_state.start_turn()
            new_state = new_state.roll_dice()

        return new_state
