"""Support for the Multi-Agent pettingzoo environment."""

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.wrappers import AssertOutOfBoundsWrapper

from farkle_ai.game import (
    Action,
    ContinueAction,
    BankAction,
    PassTurnAction,
    GameState,
    Parameters,
    scorable_patterns_by_length,
    apply_action,
    observe as observe_dict,
    to_array_observation,
    is_farkle,
    DEFAULT_SCORE_TO_WIN,
    DEFAULT_PLAYER_COUNT,
    MAX_DICE_COUNT,
    MAX_DIE_VALUE,
    MAX_PATTERN_SCORE,
)

from . import AbstractAction


ENV_NAME = "farkle_pettingzoo_v0"


def env(
    player_count: int = DEFAULT_PLAYER_COUNT,
    score_to_win: int = DEFAULT_SCORE_TO_WIN,
    render_mode=None,
):
    """Return a wrapped environment."""

    environment = FarkleEnv(
        player_count=player_count,
        score_to_win=score_to_win,
        render_mode=render_mode,
    )
    environment = AssertOutOfBoundsWrapper(environment)
    return environment


class FarkleEnv(AECEnv):
    """A raw Multi-Agent environment used for training AI agents"""

    metadata = {"render_modes": ["human"], "name": ENV_NAME}

    def __init__(
        self,
        player_count: int = DEFAULT_PLAYER_COUNT,
        score_to_win: int = DEFAULT_SCORE_TO_WIN,
        render_mode=None,
    ):
        super().__init__()

        # Farkle parameters
        self.player_count = player_count
        self.score_to_win = score_to_win
        self._state: GameState
        self.np_random: np.random.Generator

        # Agent setup
        self.possible_agents = ["player_" + str(i) for i in range(player_count)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping: dict[str, int] = dict(
            zip(self.possible_agents, range(self.player_count))
        )
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection: str = self._agent_selector.reset()

        # Petting Zoo required attributes
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos: dict = {agent: {} for agent in self.agents}
        self.observations: dict = {agent: {} for agent in self.agents}

        self._observation_space = Box(0, score_to_win + MAX_PATTERN_SCORE, (10,))
        self._action_space = Discrete(len(AbstractAction))
        self.render_mode = render_mode

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        self.np_random = np.random.default_rng(seed=seed)
        params = Parameters(
            player_count=self.player_count,
            score_to_win=self.score_to_win,
            max_dice_count=MAX_DICE_COUNT,
            max_die_value=MAX_DIE_VALUE,
        )
        self._state = GameState(parameters=params)

        # Reset Petting Zoo state
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.steps = 0

        self._state = self._state.roll_dice(self.np_random)

        while is_farkle(self._state.turn_state.rolled_dice):
            self._state = apply_action(self._state, PassTurnAction())

        # Get initial observations for all agents
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.agent_selection = self.agents[self._state.current_player]

        if self.render_mode == "human":
            self.render()

        return self.observations, self.infos

    def step(self, action: int):
        """
        Accepts and executes the action of the current agent_selection in the environment.
        Automatically switches control to the next agent.
        """

        current_agent = self.agent_selection

        self.infos[current_agent] = {}
        self._cumulative_rewards[current_agent] = 0
        self.rewards = {agent: 0 for agent in self.agents}

        if self.terminations[current_agent] or self.truncations[current_agent]:
            return self._was_dead_step(action)

        engine_action: Action
        best_patterns_by_count = scorable_patterns_by_length(
            self._state.turn_state.rolled_dice
        )

        try:
            chosen_action_enum = AbstractAction(action)
            action_type = chosen_action_enum.type
            dice_count_to_hold = chosen_action_enum.dice_count
            candidate_pattern = best_patterns_by_count[dice_count_to_hold - 1]
            is_valid_pattern_selected = (
                candidate_pattern is not None
                and candidate_pattern.score > 0
                and len(candidate_pattern.pattern) == dice_count_to_hold
            )

            if is_valid_pattern_selected:
                if action_type == "bank":
                    engine_action = BankAction(candidate_pattern.pattern)
                else:  # continue
                    engine_action = ContinueAction(candidate_pattern.pattern)

                self.infos[current_agent]["action_type"] = action_type
                self.infos[current_agent][
                    "pattern_selected"
                ] = candidate_pattern.pattern
                self.infos[current_agent][
                    "score_from_pattern"
                ] = candidate_pattern.score

                if self.render_mode == "human":
                    print(f"Action:   {action_type} - {candidate_pattern.pattern}")
                    print(f"Reward:   {candidate_pattern.score}")
            else:
                engine_action = PassTurnAction()
                self.infos[current_agent]["farkle_invalid_selection"] = True

        except ValueError:
            engine_action = PassTurnAction()
            self.infos[current_agent]["farkle_invalid_action"] = True

        while is_farkle(self._state.turn_state.rolled_dice):
            engine_action = PassTurnAction()
            self.infos[current_agent]["farkle_roll"] = True

        self._state = apply_action(self._state, engine_action)

        game_is_terminated = self._state.winner is not None

        if game_is_terminated:
            for agent in self.agents:
                self.terminations[agent] = True
                agent_index = self.agent_name_mapping[agent]
                if agent_index == self._state.winner:
                    self.rewards[agent] += 1
                    self._cumulative_rewards[agent] += 1
                    self.infos[agent]["game_outcome"] = "win"
                else:
                    self.rewards[agent] -= 1
                    self._cumulative_rewards[agent] -= 1
                    self.infos[agent]["game_outcome"] = "lose"

        if not game_is_terminated:
            while is_farkle(self._state.turn_state.rolled_dice):
                self._state = apply_action(self._state, PassTurnAction())

            while self.agent_selection != self.agents[self._state.current_player]:
                self.agent_selection = self._agent_selector.next()

        self.observations = {
            agent_id: self.observe(agent_id) for agent_id in self.agents
        }

        if self.render_mode == "human":
            self.render()

        return (
            self.observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def render(self):
        """Renders current environment state."""
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        current_agent_id = self.possible_agents[self._state.current_player]

        print(f"\n----- Current Turn {self._state.turn} ----- ")
        print(f"Current Player: {current_agent_id}")
        print(f"Player Scores: {self._state.player_scores}")
        print(f"Current Turn Score: {self._state.turn_state.score}")
        print(f"Dice Count: {self._state.turn_state.next_roll_dice_count}")
        print(f"Rolled Dice: {self._state.turn_state.rolled_dice}")
        available_patterns_by_count = scorable_patterns_by_length(
            self._state.turn_state.rolled_dice
        )
        print(
            f"Best Patterns per Dice Count: {[(p.pattern, p.score) if p else None for p in available_patterns_by_count]}"
        )
        print(f"Cumulative Reward: {self._cumulative_rewards[current_agent_id]}")
        print(f"Terminations: {self.terminations}")
        print("-" * 20)

    def observation_space(self, agent) -> Box:
        """Returns observation space for abstract farkle."""
        return self._observation_space

    def observe(self, agent) -> np.ndarray:
        """Generates the observation for a given agent."""
        agent_index = self.agent_name_mapping[agent]

        observation_dict = observe_dict(self._state, agent_index)
        observation_array = to_array_observation(observation_dict)

        return observation_array

    def action_space(self, agent) -> Discrete:
        """Returns action space for abstract farkle."""
        return self._action_space

    def state(self) -> GameState:
        """Returns current game state."""
        return self._state
