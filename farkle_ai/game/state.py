"""Defines Farkle game state."""

from dataclasses import dataclass, field
import numpy as np
from .rules import (
    contains_pattern,
    ScoringPattern,
    DEFAULT_PLAYER_COUNT,
    DEFAULT_SCORE_TO_WIN,
    MAX_DICE_COUNT,
    MAX_DIE_VALUE,
)


@dataclass(frozen=True)
class Parameters:
    """Parameters of a farkle game."""

    player_count: int = field(default=DEFAULT_PLAYER_COUNT)
    score_to_win: int = field(default=DEFAULT_SCORE_TO_WIN)
    max_dice_count: int = field(default=MAX_DICE_COUNT)
    max_die_value: int = field(default=MAX_DIE_VALUE)


@dataclass(frozen=True)
class TurnState:
    """State of the current turn."""

    next_roll_dice_count: int = field(default=6)
    score: int = field(default=0)
    rolled_dice: tuple[int, ...] = field(default_factory=tuple)
    has_rolled: bool = field(default=False)
    has_ended: bool = field(default=False)

    def select_pattern(
        self, scoring_pattern: ScoringPattern, max_dice_count=MAX_DICE_COUNT
    ) -> "TurnState":
        """Select a dice pattern."""
        if self.has_ended:
            raise RuntimeError("Can not select a pattern when turn has ended")
        if not self.has_rolled:
            raise RuntimeError("Can not select a pattern when player has not rolled")
        if len(scoring_pattern.pattern) > len(self.rolled_dice):
            raise ValueError("Selected pattern requires more dice than available")
        if not contains_pattern(self.rolled_dice, scoring_pattern.pattern):
            raise ValueError("Rolled dice do not contain selected pattern")

        remaining_dice_count = len(self.rolled_dice) - len(scoring_pattern.pattern)
        new_dice_count = (
            remaining_dice_count if remaining_dice_count > 0 else max_dice_count
        )

        new_state = TurnState(
            next_roll_dice_count=new_dice_count,
            score=self.score + scoring_pattern.score,
            rolled_dice=self.rolled_dice,
            has_rolled=False,
            has_ended=False,
        )

        return new_state

    def roll_dice(
        self,
        rng=None,
        max_dice_count=MAX_DICE_COUNT,
        max_die_value=MAX_DIE_VALUE,
    ) -> "TurnState":
        """Roll the dice."""
        if self.has_ended:
            raise RuntimeError("Can not roll dice when turn has ended")
        if self.has_rolled:
            raise RuntimeError("Can not roll dice when dice are already rolled")

        if rng is None:
            rng = np.random.default_rng()

        new_dice_count = (
            max_dice_count
            if self.next_roll_dice_count == 0
            else self.next_roll_dice_count
        )

        new_rolled_dice = tuple(
            rng.choice(
                np.arange(1, max_die_value + 1),
                size=new_dice_count,
                replace=True,
            )
        )

        new_state = TurnState(
            new_dice_count,
            self.score,
            new_rolled_dice,
            True,
        )

        return new_state

    def pass_turn(self) -> "TurnState":
        """Pass current turn."""
        if self.has_ended:
            raise RuntimeError("Can not pass an ended turn")

        new_state = TurnState(
            next_roll_dice_count=0,
            score=0,
            rolled_dice=tuple(),
            has_rolled=False,
            has_ended=True,
        )

        return new_state

    def end_turn(self) -> "TurnState":
        """End current turn."""
        if self.has_ended:
            raise RuntimeError("Can not end an already ended turn")
        if self.has_rolled:
            raise RuntimeError("Can not end turn after rolling")

        new_state = TurnState(
            next_roll_dice_count=0,
            score=self.score,
            rolled_dice=tuple(),
            has_rolled=False,
            has_ended=True,
        )

        return new_state


@dataclass(frozen=True)
class GameState:
    """State of farkle game."""

    parameters: Parameters
    turn_state: TurnState
    player_scores: list[int]
    current_player: int
    winner: int | None
    turn: int
    is_quit: bool

    def __init__(
        self,
        parameters: Parameters | None = None,
        turn_state: TurnState | None = None,
        player_scores: list[int] | None = None,
        current_player: int = 0,
        winner: int | None = None,
        turn: int = 0,
        is_quit: bool = False,
    ):

        object.__setattr__(
            self, "parameters", parameters if parameters is not None else Parameters()
        )
        object.__setattr__(
            self,
            "turn_state",
            (
                turn_state
                if turn_state is not None
                else TurnState(self.parameters.max_dice_count)
            ),
        )
        object.__setattr__(
            self,
            "player_scores",
            (
                player_scores
                if player_scores is not None
                else [0 for _ in range(self.parameters.player_count)]
            ),
        )
        object.__setattr__(self, "current_player", current_player)
        object.__setattr__(self, "winner", winner)
        object.__setattr__(self, "turn", turn)
        object.__setattr__(self, "is_quit", is_quit)

    def select_pattern(self, scoring_pattern: ScoringPattern) -> "GameState":
        """Select dice pattern and continue the round."""

        new_state = GameState(
            self.parameters,
            self.turn_state.select_pattern(scoring_pattern),
            self.player_scores,
            self.current_player,
            self.winner,
            self.turn,
        )

        return new_state

    def roll_dice(self, rng=None) -> "GameState":
        """Rolls the dice."""

        new_state = GameState(
            self.parameters,
            self.turn_state.roll_dice(
                rng, self.parameters.max_dice_count, self.parameters.max_die_value
            ),
            self.player_scores,
            self.current_player,
            self.winner,
            self.turn,
        )

        return new_state

    def pass_turn(self) -> "GameState":
        """Pass current turn."""

        new_state = GameState(
            parameters=self.parameters,
            turn_state=self.turn_state.pass_turn(),
            current_player=self.current_player,
            player_scores=self.player_scores,
            winner=self.winner,
            turn=self.turn,
        )

        return new_state

    def end_turn(self) -> "GameState":
        """End current turn."""

        new_player_scores = list(self.player_scores)
        new_player_scores[self.current_player] += self.turn_state.score
        new_winner = (
            self.current_player
            if new_player_scores[self.current_player] >= self.parameters.score_to_win
            else None
        )

        new_state = GameState(
            parameters=self.parameters,
            turn_state=self.turn_state.end_turn(),
            current_player=self.current_player,
            player_scores=new_player_scores,
            winner=new_winner,
            turn=self.turn,
        )

        return new_state

    def start_turn(self, max_dice_count=MAX_DICE_COUNT) -> "GameState":
        """Start a new turn."""

        new_turn_state = TurnState(
            next_roll_dice_count=max_dice_count,
            score=0,
            rolled_dice=tuple(),
            has_rolled=False,
            has_ended=False,
        )

        new_current_player = (self.current_player + 1) % self.parameters.player_count

        new_state = GameState(
            parameters=self.parameters,
            turn_state=new_turn_state,
            current_player=new_current_player,
            player_scores=self.player_scores,
            winner=self.winner,
            turn=self.turn + 1 if new_current_player == 0 else self.turn,
        )

        return new_state
