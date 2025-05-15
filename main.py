# main.py

import time
from collections import Counter
import pygame
import numpy as np

from farkle_ai.agents import (
    AgentWrapper,
    NeuralNetworkAgent,
    NaiveAgent,
    RandomAgent,
    BankHighestAgent,
)
from farkle_ai.game import (
    GameState,
    Parameters,
    Action,
    ContinueAction,
    BankAction,
    PassTurnAction,
    pattern_score,
    apply_action,
    is_farkle,
)

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
BACKGROUND_COLOR = (50, 50, 50)
TEXT_COLOR = (220, 220, 220)
PLAYER_COLORS = [(100, 150, 255), (255, 100, 150), (100, 255, 100), (255, 150, 100)]
DIE_COLOR = (200, 200, 200)
DIE_DOT_COLOR = (30, 30, 30)
DIE_SELECTED_BORDER_COLOR = (255, 255, 0)
DIE_AI_HIGHLIGHT_COLOR = (0, 255, 0)
BUTTON_COLOR = (70, 70, 90)
BUTTON_HOVER_COLOR = (100, 100, 120)
BUTTON_DISABLED_COLOR = (40, 40, 50)
FARKLE_COLOR = (255, 50, 50)
AI_ACTION_TEXT_COLOR = (200, 200, 0)

FPS = 30
DIE_SIZE = 50
DIE_PADDING = 10

PLAYER_HUMAN = "human"
PLAYER_AI = "ai"

BUTTON_BANK_ID = "bank_selected"
BUTTON_ROLL_ID = "roll_remaining"
BUTTON_CLEAR_ID = "clear_selection"


AVAILABLE_AIS = {
    "1": {"name": "Neural Network", "class": NeuralNetworkAgent},
    "2": {"name": "Naive", "class": NaiveAgent},
    "3": {"name": "Bank", "class": BankHighestAgent},
    "4": {"name": "Random", "class": RandomAgent},
}


def _find_indices_for_pattern(
    rolled_dice: tuple[int, ...], pattern_to_find: tuple[int, ...]
) -> list[int]:
    if not pattern_to_find:
        return []
    pattern_counts = Counter(pattern_to_find)
    found_indices = []
    temp_rolled_dice_indices_available = list(range(len(rolled_dice)))
    for p_val_needed in pattern_to_find:
        found_this_val = False
        for i, r_idx in enumerate(temp_rolled_dice_indices_available):
            if r_idx == -1:
                continue
            if rolled_dice[r_idx] == p_val_needed:
                found_indices.append(r_idx)
                temp_rolled_dice_indices_available[i] = -1
                found_this_val = True
                break
        if not found_this_val:
            return []
    return sorted(found_indices)


class FarklePygameGame:
    def __init__(
        self,
        game_mode="pvp",
        num_players=2,
        score_to_win=5000,
        ai_model_classes: list | None = None,
    ):
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Farkle")
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.SysFont("arial", 20)
        self.font_medium = pygame.font.SysFont("arial", 30)
        self.font_large = pygame.font.SysFont("arial", 48)

        self.rng = np.random.default_rng()

        self.initial_game_config = {
            "game_mode": game_mode,
            "num_players": num_players,
            "score_to_win": score_to_win,
            "ai_model_classes": (ai_model_classes if ai_model_classes else []),
        }

        self.params = Parameters(player_count=num_players, score_to_win=score_to_win)

        self.player_types = []
        if game_mode == "pvp":
            self.player_types = [PLAYER_HUMAN] * num_players
        elif game_mode == "pva":
            if num_players < 2:
                num_players = 2
                self.params = Parameters(
                    player_count=2, score_to_win=score_to_win
                )  # Adjust params
            self.player_types = [PLAYER_HUMAN] + [PLAYER_AI] * (num_players - 1)
        elif game_mode == "ava":
            self.player_types = [PLAYER_AI] * num_players
        else:
            self.player_types = [PLAYER_HUMAN] * num_players

        # Recalculate num_players if mode changed it implicitly, and update params
        if len(self.player_types) != self.params.player_count:
            self.params = Parameters(
                player_count=len(self.player_types), score_to_win=score_to_win
            )
            self.initial_game_config["num_players"] = len(
                self.player_types
            )  # Update stored config

        self.game_state = GameState(parameters=self.params, current_player=-1)
        self.game_state = self.game_state.start_turn()
        self.game_state = self.game_state.roll_dice(self.rng)

        self.ai_agents = {}  # player_index -> AI_instance
        ai_class_idx = 0
        current_ai_model_classes = self.initial_game_config["ai_model_classes"]
        for i, player_type in enumerate(self.player_types):
            if player_type == PLAYER_AI:
                if current_ai_model_classes and ai_class_idx < len(
                    current_ai_model_classes
                ):
                    chosen_ai_class = current_ai_model_classes[ai_class_idx]
                    self.ai_agents[i] = AgentWrapper(chosen_ai_class())  # Instantiate
                    ai_class_idx += 1
                else:  # Fallback if not enough classes provided or none given
                    self.ai_agents[i] = AgentWrapper(NaiveAgent())

        self.human_player_action_buttons = []
        self.selected_dice_indices = []
        self.die_rects = []
        self.highlighted_ai_choice_indices = []
        self.ai_highlight_end_time = 0
        self.ai_highlight_duration = 2.0
        self.message_text = ""
        self.message_time = 0
        self.message_color = TEXT_COLOR
        self.running = True

    def _draw_text(self, text, font, color, x, y, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        self.screen.blit(text_surface, text_rect)

    def _draw_dice(self, dice_values, x_start, y_pos):
        self.die_rects.clear()
        current_time = time.time()
        is_ai_highlighting = (
            self.highlighted_ai_choice_indices
            and current_time < self.ai_highlight_end_time
        )

        for i, value in enumerate(dice_values):
            rect = pygame.Rect(
                x_start + i * (DIE_SIZE + DIE_PADDING), y_pos, DIE_SIZE, DIE_SIZE
            )
            self.die_rects.append(rect)

            border_color = None
            fill_color = DIE_COLOR
            if i in self.selected_dice_indices:
                border_color = DIE_SELECTED_BORDER_COLOR
            elif is_ai_highlighting and i in self.highlighted_ai_choice_indices:
                fill_color = DIE_AI_HIGHLIGHT_COLOR
            pygame.draw.rect(self.screen, fill_color, rect, border_radius=5)
            if border_color:
                pygame.draw.rect(
                    self.screen, border_color, rect, width=3, border_radius=5
                )
            self._draw_text(
                str(value),
                self.font_medium,
                DIE_DOT_COLOR,
                rect.centerx,
                rect.centery,
                center=True,
            )

    def _get_current_selected_pattern_and_score(self):
        if not self.selected_dice_indices:
            return None, 0
        selected_dice_values = [
            self.game_state.turn_state.rolled_dice[idx]
            for idx in self.selected_dice_indices
            if 0 <= idx < len(self.game_state.turn_state.rolled_dice)
        ]
        if not selected_dice_values:
            return None, 0
        return tuple(sorted(selected_dice_values)), pattern_score(
            tuple(sorted(selected_dice_values))
        )

    def _draw_human_player_buttons(self):
        self.human_player_action_buttons.clear()
        start_y = SCREEN_HEIGHT - 120
        button_height = 40
        button_width = 200
        padding = 20
        _, current_selection_score = self._get_current_selected_pattern_and_score()
        selection_is_valid = current_selection_score > 0
        buttons_to_draw = []
        if self.selected_dice_indices:
            buttons_to_draw.append(
                {"id": BUTTON_CLEAR_ID, "text": "Clear", "enabled": True}
            )
        buttons_to_draw.append(
            {
                "id": BUTTON_BANK_ID,
                "text": f"Bank ({current_selection_score if selection_is_valid else 0})",
                "enabled": selection_is_valid,
            }
        )
        buttons_to_draw.append(
            {
                "id": BUTTON_ROLL_ID,
                "text": f"Roll ({current_selection_score if selection_is_valid else 0})",
                "enabled": selection_is_valid,
            }
        )
        total_width = (
            len(buttons_to_draw) * button_width + (len(buttons_to_draw) - 1) * padding
        )
        start_x = (SCREEN_WIDTH - total_width) // 2
        for i, btn_info in enumerate(buttons_to_draw):
            x_pos = start_x + i * (button_width + padding)
            rect = pygame.Rect(x_pos, start_y, button_width, button_height)
            mouse_pos = pygame.mouse.get_pos()
            color = (
                BUTTON_DISABLED_COLOR
                if not btn_info["enabled"]
                else (
                    BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) else BUTTON_COLOR
                )
            )
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            self._draw_text(
                btn_info["text"],
                self.font_small,
                TEXT_COLOR,
                rect.centerx,
                rect.centery,
                center=True,
            )
            self.human_player_action_buttons.append(
                {"rect": rect, "id": btn_info["id"], "enabled": btn_info["enabled"]}
            )

    def _display_message(
        self, text, duration_seconds=2000, color=TEXT_COLOR, font=None
    ):
        self.message_text = text
        self.message_time = time.time() + duration_seconds
        self.message_color = color
        self.message_font = font if font else self.font_large

    def _handle_human_input(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, die_rect in enumerate(self.die_rects):
                if die_rect.collidepoint(event.pos):
                    if i in self.selected_dice_indices:
                        self.selected_dice_indices.remove(i)
                    else:
                        self.selected_dice_indices.append(i)
                    self.selected_dice_indices.sort()
                    return True
            for button_info in self.human_player_action_buttons:
                if button_info["enabled"] and button_info["rect"].collidepoint(
                    event.pos
                ):
                    action_id = button_info["id"]
                    if action_id == BUTTON_CLEAR_ID:
                        self.selected_dice_indices.clear()
                        return True
                    selected_pattern, score = (
                        self._get_current_selected_pattern_and_score()
                    )
                    if selected_pattern and score > 0:
                        action_to_take = (
                            BankAction(selected_pattern)
                            if action_id == BUTTON_BANK_ID
                            else (
                                ContinueAction(selected_pattern)
                                if action_id == BUTTON_ROLL_ID
                                else None
                            )
                        )
                        if action_to_take:
                            self.game_state = apply_action(
                                self.game_state, action_to_take, self.rng
                            )
                            self.selected_dice_indices.clear()
                            return True
                    return True
        return False

    def _prepare_ai_highlight(self, ai_chosen_pattern: tuple[int, ...]):
        self.highlighted_ai_choice_indices = _find_indices_for_pattern(
            self.game_state.turn_state.rolled_dice, ai_chosen_pattern
        )
        self.ai_highlight_end_time = time.time() + self.ai_highlight_duration

    def _render(self):
        self.screen.fill(BACKGROUND_COLOR)
        score_y_pos = 20
        for i in range(self.params.player_count):
            player_name_suffix = ""
            if self.player_types[i] == PLAYER_AI:
                ai_instance = self.ai_agents.get(i)
                ai_type_name = str(ai_instance) if ai_instance else "AI"
                player_name_suffix = f" ({ai_type_name})"  # Show AI type
            player_text = f"P{i+1}{player_name_suffix}"
            score_text = f"{player_text}: {self.game_state.player_scores[i]}"
            color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
            prefix = "> " if i == self.game_state.current_player else ""
            self._draw_text(
                prefix + score_text,
                self.font_medium,
                color if i == self.game_state.current_player else TEXT_COLOR,
                20,
                score_y_pos + i * 35,
            )

        self._draw_text(
            f"Turn: {self.game_state.turn + 1}",
            self.font_small,
            TEXT_COLOR,
            SCREEN_WIDTH - 175,
            20,
        )
        self._draw_text(
            f"Target: {self.params.score_to_win}",
            self.font_small,
            TEXT_COLOR,
            SCREEN_WIDTH - 175,
            45,
        )
        self._draw_text(
            f"Player {self.game_state.current_player + 1}'s Turn",
            self.font_medium,
            TEXT_COLOR,
            SCREEN_WIDTH // 2,
            80 + 100,
            center=True,
        )
        self._draw_text(
            f"Turn Score: {self.game_state.turn_state.score}",
            self.font_medium,
            TEXT_COLOR,
            SCREEN_WIDTH // 2,
            120 + 100,
            center=True,
        )

        if self.game_state.turn_state.has_rolled:
            self._draw_text(
                "Rolled Dice:",
                self.font_medium,
                TEXT_COLOR,
                SCREEN_WIDTH // 2,
                180 + 100,
                center=True,
            )
            num_dice = len(self.game_state.turn_state.rolled_dice)
            dice_area_width = num_dice * DIE_SIZE + (num_dice - 1) * DIE_PADDING
            dice_x_start = (SCREEN_WIDTH - dice_area_width) // 2
            self._draw_dice(
                self.game_state.turn_state.rolled_dice, dice_x_start, 220 + 100
            )
            if self.player_types[self.game_state.current_player] == PLAYER_HUMAN:
                _, selection_score = self._get_current_selected_pattern_and_score()
                self._draw_text(
                    f"Selected Value: {selection_score}",
                    self.font_small,
                    TEXT_COLOR,
                    SCREEN_WIDTH // 2,
                    220 + DIE_SIZE + 15 + 100,
                    center=True,
                )

        if (
            self.player_types[self.game_state.current_player] == PLAYER_HUMAN
            and self.game_state.turn_state.has_rolled
            and not is_farkle(self.game_state.turn_state.rolled_dice)
        ):
            self._draw_human_player_buttons()

        current_time = time.time()
        if self.message_text and current_time < self.message_time:
            font_to_use = getattr(self, "message_font", self.font_large)
            self._draw_text(
                self.message_text,
                font_to_use,
                self.message_color,
                SCREEN_WIDTH // 2,
                SCREEN_HEIGHT // 2 + 40 + 100,
                center=True,
            )
        elif self.message_text and current_time >= self.message_time:
            self.message_text = ""

        if self.game_state.winner is not None:
            winner_text = f"Player {self.game_state.winner + 1} Wins!"
            self._draw_text(
                winner_text,
                self.font_large,
                (255, 215, 0),
                SCREEN_WIDTH // 2,
                SCREEN_HEIGHT - 80,
                center=True,
            )
            self._draw_text(
                "R to Restart, Q to Quit",
                self.font_small,
                TEXT_COLOR,
                SCREEN_WIDTH // 2,
                SCREEN_HEIGHT - 40,
                center=True,
            )
            self.highlighted_ai_choice_indices.clear()
        pygame.display.flip()

    def _game_over_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                # Re-initialize with stored config
                self.__init__(**self.initial_game_config)
                return True
            if event.key == pygame.K_q:
                self.running = False
                return True
        return False

    def run(self):
        action_processed_this_turn_segment = False
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if self.game_state.winner is not None:
                    if self._game_over_input(event):
                        action_processed_this_turn_segment = True
                        break
                elif (
                    self.player_types[self.game_state.current_player] == PLAYER_HUMAN
                    and self.game_state.turn_state.has_rolled
                    and not is_farkle(self.game_state.turn_state.rolled_dice)
                ):
                    if self._handle_human_input(event):
                        action_processed_this_turn_segment = True
                        break

            if not self.running:
                break
            if action_processed_this_turn_segment and self.game_state.winner is None:
                if not (
                    self.game_state.turn_state.has_rolled
                    and not is_farkle(self.game_state.turn_state.rolled_dice)
                    and self.player_types[self.game_state.current_player]
                    == PLAYER_HUMAN
                ):
                    action_processed_this_turn_segment = False

            if self.game_state.winner is not None:
                self._render()
                self.clock.tick(FPS)
                continue

            current_player_idx = self.game_state.current_player
            current_player_type = self.player_types[current_player_idx]

            if (
                self.game_state.turn_state.has_rolled
                and not action_processed_this_turn_segment
            ):
                if is_farkle(self.game_state.turn_state.rolled_dice):
                    self.selected_dice_indices.clear()
                    self.highlighted_ai_choice_indices.clear()
                    self._display_message(
                        "Farkle!", 2, FARKLE_COLOR, font=self.font_large
                    )
                    self._render()
                    pygame.time.wait(2000)
                    self.game_state = apply_action(
                        self.game_state, PassTurnAction(), self.rng
                    )
                    action_processed_this_turn_segment = False

                elif current_player_type == PLAYER_AI:
                    if time.time() > self.ai_highlight_end_time:
                        self.highlighted_ai_choice_indices.clear()
                        self._render()
                        pygame.time.wait(1000)

                        current_ai_agent = self.ai_agents[
                            current_player_idx
                        ]  # Get specific AI agent
                        ai_action = current_ai_agent.compute_action(self.game_state)

                        ai_action_text = ""
                        if isinstance(ai_action, ContinueAction):
                            ai_action_text = "AI Rolls"
                        elif isinstance(ai_action, BankAction):
                            ai_action_text = "AI Banks"

                        if hasattr(ai_action, "pattern") and ai_action.pattern:
                            self._prepare_ai_highlight(ai_action.pattern)
                            if ai_action_text:
                                self._display_message(
                                    ai_action_text,
                                    self.ai_highlight_duration,
                                    AI_ACTION_TEXT_COLOR,
                                    font=self.font_medium,
                                )
                            self._render()
                            pygame.time.wait(int(self.ai_highlight_duration * 1000))

                        self.game_state = apply_action(
                            self.game_state, ai_action, self.rng
                        )
                        self.highlighted_ai_choice_indices.clear()
                        action_processed_this_turn_segment = False
            self._render()
            self.clock.tick(FPS)
        pygame.quit()


if __name__ == "__main__":
    print("--- Welcome to Farkle! ---")
    print("Choose game mode:")
    print("1: Player vs Player")
    print("2: Player vs AI")
    print("3: AI vs AI")

    mode_choice = ""
    while mode_choice not in ["1", "2", "3"]:
        mode_choice = input("Enter mode choice (1-3): ")

    game_mode_str = "pvp"
    num_players_choice = 2  # Default

    if mode_choice == "1":
        game_mode_str = "pvp"
    elif mode_choice == "2":
        game_mode_str = "pva"
    elif mode_choice == "3":
        game_mode_str = "ava"

    if game_mode_str == "pvp" or game_mode_str == "pva" or game_mode_str == "ava":
        num_players_choice = 2

    chosen_ai_model_classes = []
    num_ai_to_select = 0
    if game_mode_str == "pva":
        num_ai_to_select = 1
    elif game_mode_str == "ava":
        num_ai_to_select = num_players_choice  # All players are AI

    if num_ai_to_select > 0:
        print("\n--- Select AI Model(s) ---")
        for i in range(num_ai_to_select):
            print(f"\nChoose AI for AI Player {i+1}:")
            for key, ai_info in AVAILABLE_AIS.items():
                print(f"  {key}: {ai_info['name']}")

            ai_choice_key = ""
            while ai_choice_key not in AVAILABLE_AIS:
                ai_choice_key = input(
                    f"Select AI model for AI Player {i+1} (e.g., 1): "
                )
            chosen_ai_model_classes.append(AVAILABLE_AIS[ai_choice_key]["class"])

    score_to_win_choice = 5000
    try:
        raw_score = input(f"\nEnter score to win (default: {score_to_win_choice}): ")
        if raw_score.strip():
            score_to_win_choice = int(raw_score)
    except ValueError:
        print(f"Invalid score, using default {score_to_win_choice}.")

    print("\nStarting game...")
    game = FarklePygameGame(
        game_mode=game_mode_str,
        num_players=num_players_choice,
        score_to_win=score_to_win_choice,
        ai_model_classes=chosen_ai_model_classes,
    )
    game.run()
