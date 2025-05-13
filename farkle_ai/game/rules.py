"""Define rules and scoring for the farkle dice game."""

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cache

# Default game settings
DEFAULT_SCORE_TO_WIN = 5000
MAX_DICE_COUNT = 6
MAX_DIE_VALUE = 6
DEFAULT_PLAYER_COUNT = 2

# Scoring values
SCORE_ONE = 100
SCORE_FIVE = 50
SCORE_TRIPLE_ONES_MULTIPLIER = 1000
SCORE_TRIPLE_MULTIPLIER = 100
SCORE_ABOVE_TRIPLE_MULTIPLIER = 2


@dataclass
class ScoringPattern:
    """A scoring dice pattern."""

    pattern: tuple[int, ...]
    score: int

    def __add__(self, other: "ScoringPattern") -> "ScoringPattern":
        pattern_counter = Counter(self.pattern) + Counter(other.pattern)
        pattern = tuple(sorted(pattern_counter.elements()))

        return ScoringPattern(
            pattern,
            self.score + other.score,
        )


MAX_PATTERN_SCORE = 8000

BASE_SCORING_PATTERNS: list[ScoringPattern] = [
    # Six of a kind
    ScoringPattern((1, 1, 1, 1, 1, 1), 8000),
    ScoringPattern((2, 2, 2, 2, 2, 2), 1600),
    ScoringPattern((3, 3, 3, 3, 3, 3), 2400),
    ScoringPattern((4, 4, 4, 4, 4, 4), 3200),
    ScoringPattern((5, 5, 5, 5, 5, 5), 4000),
    ScoringPattern((6, 6, 6, 6, 6, 6), 4800),
    # Five of a kind
    ScoringPattern((1, 1, 1, 1, 1), 4000),
    ScoringPattern((2, 2, 2, 2, 2), 800),
    ScoringPattern((3, 3, 3, 3, 3), 1200),
    ScoringPattern((4, 4, 4, 4, 4), 1600),
    ScoringPattern((5, 5, 5, 5, 5), 2000),
    ScoringPattern((6, 6, 6, 6, 6), 2400),
    # Four of a kind
    ScoringPattern((1, 1, 1, 1), 2000),
    ScoringPattern((2, 2, 2, 2), 400),
    ScoringPattern((3, 3, 3, 3), 600),
    ScoringPattern((4, 4, 4, 4), 800),
    ScoringPattern((5, 5, 5, 5), 1000),
    ScoringPattern((6, 6, 6, 6), 1200),
    # Straights
    ScoringPattern((1, 2, 3, 4, 5, 6), 1500),
    ScoringPattern((1, 2, 3, 4, 5), 500),
    ScoringPattern((2, 3, 4, 5, 6), 750),
    # Three of a kind
    ScoringPattern((1, 1, 1), 1000),
    ScoringPattern((2, 2, 2), 200),
    ScoringPattern((3, 3, 3), 300),
    ScoringPattern((4, 4, 4), 400),
    ScoringPattern((5, 5, 5), 500),
    ScoringPattern((6, 6, 6), 600),
    # Single Dice
    ScoringPattern((1,), 100),
    ScoringPattern((5,), 50),
]

# Helpers


def roll_dice(dice_count: int = 6, dice_max_value: int = 6) -> list[int]:
    """Rolls a specified number of dice."""
    if dice_count <= 0:
        return []
    return random.choices(range(1, dice_max_value + 1), k=dice_count)


def subtract_patterns(
    rolled_dice: tuple[int, ...], pattern: tuple[int, ...]
) -> Counter:
    """Subtracts a given pattern from the rolled dice. Creates a copy."""
    a = Counter(rolled_dice)
    b = Counter(pattern)
    a.subtract(b)
    return a


def contains_pattern(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    """Checks whether dice pattern a contains dice pattern b."""
    remainder = subtract_patterns(a, b)
    return are_dice_valid(remainder)


def are_dice_valid(dice: Counter) -> bool:
    """Checks if the dice Counter object does not contain negative values."""
    return all((value >= 0 for value in dice.values()))


def are_dice_zero(dice: Counter) -> bool:
    """Checks if the dice Counter object only contains zeroe values."""
    return all((value == 0 for value in dice.values()))


@cache
def scorable_patterns_table() -> dict[tuple[int, ...], int]:
    """Returns a dict of all possible scorable patterns in the format (pattern: score)."""

    dp: dict[int, dict[tuple[int, ...], int]] = defaultdict(dict)

    for scoring_pattern in BASE_SCORING_PATTERNS:
        count = len(scoring_pattern.pattern)
        pattern = scoring_pattern.pattern
        score = scoring_pattern.score
        dp[count][pattern] = score

    for count in range(1, MAX_DICE_COUNT + 1):
        for count_a in range(1, count // 2 + 1):
            count_b = count - count_a
            for pattern_a, score_a in dp[count_a].items():
                for pattern_b, score_b in dp[count_b].items():
                    pattern = tuple(sorted(pattern_a + pattern_b))
                    score = score_a + score_b
                    if score > dp[count].get(pattern, 0):
                        dp[count][pattern] = score

    return {
        pattern: score for scores in dp.values() for pattern, score in scores.items()
    }


def pattern_score(dice: tuple[int, ...]) -> int:
    """Return potential awarded score for a given pattern."""
    return scorable_patterns_table().get(dice, 0)


@cache
def scorable_patterns(dice: tuple[int, ...]) -> list[ScoringPattern]:
    """Finds possible scoring patterns for a given dice state."""

    patterns = [
        ScoringPattern(pattern, score)
        for pattern, score in scorable_patterns_table().items()
        if contains_pattern(dice, pattern)
    ]

    return patterns


@cache
def scorable_patterns_by_length(dice: tuple[int, ...]) -> list[ScoringPattern]:
    """Finds the best scorable pattern for each pattern length."""

    by_length = [ScoringPattern(tuple(), 0) for _ in range(MAX_DICE_COUNT)]

    for scoring_pattern in scorable_patterns(dice):
        if scoring_pattern.score > by_length[len(scoring_pattern.pattern)].score:
            by_length[len(scoring_pattern.pattern)] = scoring_pattern

    return by_length
