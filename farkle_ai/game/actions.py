"""Define possible farkle game actions."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Action:
    """Base class for all game actions."""


@dataclass(frozen=True)
class ContinueAction(Action):
    """
    Select a specific set of scoring dice from the current roll
    and continue the current turn.
    """

    pattern: tuple[int, ...] = field()


@dataclass(frozen=True)
class BankAction(Action):
    """
    Select a specific set of scoring dice from their current roll
    and bank the current turn score.
    """

    pattern: tuple[int, ...] = field()


@dataclass(frozen=True)
class PassTurnAction(Action):
    """Explicitly pass the turn, typically after a Farkle."""


@dataclass(frozen=True)
class QuitAction(Action):
    """Represents a player's decision to quit the game."""
