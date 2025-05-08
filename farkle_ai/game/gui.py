"""Support for GUI for the Farkle game."""

import tkinter as tk
from tkinter import messagebox
from .state import GameState
from .engine import FarkleEngine
from .actions import Action, ContinueAction, BankAction, PassTurnAction, QuitAction
from .rules import scoring_patterns_for_roll


DIE_SYMBOLS = {
    1: "⚀",
    2: "⚁",
    3: "⚂",
    4: "⚃",
    5: "⚄",
    6: "⚅",
}


class FarkleGUI:
    """A simple Tkinter GUI for the Farkle game."""

    def __init__(self, master: tk.Tk, initial_state: GameState, engine: FarkleEngine):
        self.master = master
        master.title("Farkle Game")

        self.state = initial_state
        self.engine = engine

        self.selected_dice_indices: list[int] = (
            []
        )  # Track indices of currently selected dice in the roll

        # --- Create Widgets ---
        self.widgets: dict = {}  # Dictionary to hold references to widgets

        # Game Status Frame
        self.widgets["status_frame"] = tk.Frame(master)
        self.widgets["status_frame"].pack(pady=10)
        self.widgets["game_status_label"] = tk.Label(
            self.widgets["status_frame"],
            text="Welcome to Farkle!",
            font=("Arial", 14, "bold"),
        )
        self.widgets["game_status_label"].pack()

        # Scores Frame
        self.widgets["scores_frame"] = tk.Frame(master)
        self.widgets["scores_frame"].pack(pady=5)
        self.widgets["score_labels"] = []
        for i in range(self.state.parameters.player_count):
            lbl = tk.Label(
                self.widgets["scores_frame"], text=f"P{i+1}: 0", font=("Arial", 12)
            )
            lbl.pack(side=tk.LEFT, padx=10)
            self.widgets["score_labels"].append(lbl)

        # Current Player and Turn Score Frame
        self.widgets["turn_info_frame"] = tk.Frame(master)
        self.widgets["turn_info_frame"].pack(pady=5)
        self.widgets["current_player_label"] = tk.Label(
            self.widgets["turn_info_frame"],
            text="Player 1's Turn",
            font=("Arial", 12, "italic"),
        )
        self.widgets["current_player_label"].pack(side=tk.LEFT, padx=10)
        self.widgets["turn_score_label"] = tk.Label(
            self.widgets["turn_info_frame"], text="Turn Score: 0", font=("Arial", 12)
        )
        self.widgets["turn_score_label"].pack(side=tk.LEFT, padx=10)

        # Rolled Dice Frame
        self.widgets["dice_frame"] = tk.Frame(master)
        self.widgets["dice_frame"].pack(pady=10)
        self.widgets["dice_labels"] = []  # Labels for individual dice

        # Actions Frame
        self.widgets["actions_frame"] = tk.Frame(master)
        self.widgets["actions_frame"].pack(pady=10)

        # Dynamic Action Buttons Frame (will be cleared and repopulated)
        self.widgets["action_buttons_frame"] = tk.Frame(self.widgets["actions_frame"])
        self.widgets["action_buttons_frame"].pack()

        # Message Label
        self.widgets["message_label"] = tk.Label(
            master, text="", fg="red", font=("Arial", 10)
        )
        self.widgets["message_label"].pack(pady=5)

        # Quit Button (always available)
        self.widgets["quit_button"] = tk.Button(
            master, text="Quit", command=self.handle_quit
        )
        self.widgets["quit_button"].pack(pady=10)

        # --- Initial Setup ---
        self.start_game()  # Start the game flow

    def clear_frame(self, frame: tk.Frame):
        """Clears all widgets from a frame."""
        for widget in frame.winfo_children():
            widget.destroy()

    def update_display(self):
        """Updates all GUI elements based on the current state."""
        state = self.state
        turn_state = state.turn_state

        # Update Status and Scores
        self.widgets["game_status_label"].config(text="Farkle Game")
        for i, score_label in enumerate(self.widgets["score_labels"]):
            score_label.config(text=f"P{i+1}: {state.player_scores[i]}")
            score_label.config(
                font=("Arial", 12, "bold" if i == state.current_player else "normal")
            )  # Highlight current player

        self.widgets["current_player_label"].config(
            text=f"Player {state.current_player + 1}'s Turn"
        )
        self.widgets["turn_score_label"].config(text=f"Turn Score: {turn_state.score}")

        # Update Rolled Dice Display
        self.clear_frame(self.widgets["dice_frame"])
        self.widgets["dice_labels"] = []  # Reset the list of dice labels
        self.selected_dice_indices = []  # Clear selections

        if turn_state.has_rolled:
            for i, die_value in enumerate(turn_state.rolled_dice):
                die_label = tk.Label(
                    self.widgets["dice_frame"],
                    text=DIE_SYMBOLS.get(
                        die_value, str(die_value)
                    ),  # Use symbol or number
                    font=("Arial", 30),
                    padx=10,
                    pady=5,
                    relief=tk.RAISED,  # Raised look initially
                    borderwidth=2,
                    bg="white",  # Default background
                )
                die_label.bind(
                    "<Button-1>", lambda event, index=i: self.on_die_click(index)
                )  # Bind click event
                die_label.pack(side=tk.LEFT, padx=5)
                self.widgets["dice_labels"].append(die_label)
            self.widgets["message_label"].config(text="")  # Clear message unless Farkle

        elif not turn_state.has_ended and turn_state.score > 0:
            # State after selecting pattern, before rolling/banking
            tk.Label(
                self.widgets["dice_frame"],
                text="Dice selected, ready for next roll or bank.",
                font=("Arial", 12),
            ).pack()
            self.widgets["message_label"].config(text="")
        else:
            # Start of turn, or after turn ended
            tk.Label(
                self.widgets["dice_frame"],
                text="Roll the dice to start the turn.",
                font=("Arial", 12),
            ).pack()
            self.widgets["message_label"].config(text="")

        # Update Action Buttons
        self.clear_frame(self.widgets["action_buttons_frame"])
        available_actions = self.engine.actions(state)

        # Handle Farkle state display
        if len(available_actions) == 2:  # Check both flags to confirm farkle *outcome*
            self.widgets["message_label"].config(
                text="FARKLE! Score for this turn is 0.", fg="red"
            )

        else:  # Not a Farkle (or not yet rolled/farkle checked)
            # If player has just rolled and there are scoring options, show action buttons
            if turn_state.has_rolled:
                scoring_patterns = scoring_patterns_for_roll(
                    turn_state.rolled_dice, self.engine.pattern_scores
                )
                if scoring_patterns:  # Only show Continue/Bank if scoring is possible
                    # Continue/Bank buttons will be enabled/disabled by check_selected_pattern
                    self.widgets["continue_button"] = tk.Button(
                        self.widgets["action_buttons_frame"],
                        text="Continue (Select Dice)",
                        command=self.handle_continue,
                        state=tk.DISABLED,  # Initially disabled
                    )
                    self.widgets["continue_button"].pack(side=tk.LEFT, padx=5)

                    self.widgets["bank_button"] = tk.Button(
                        self.widgets["action_buttons_frame"],
                        text="Bank (Select Dice)",
                        command=self.handle_bank,
                        state=tk.DISABLED,  # Initially disabled
                    )
                    self.widgets["bank_button"].pack(side=tk.LEFT, padx=5)

                # Pass button is available after any roll (farkle or not, to forfeit)
                self.widgets["pass_button"] = tk.Button(
                    self.widgets["action_buttons_frame"],
                    text="Pass Turn",
                    command=self.handle_pass,
                )
                self.widgets["pass_button"].pack(side=tk.LEFT, padx=5)

        # Re-evaluate action buttons based on state and selected dice
        self.clear_frame(self.widgets["action_buttons_frame"])
        available_actions = self.engine.actions(state)  # Potential actions from engine

        if len(available_actions) == 2:
            # On Farkle, the only effective action is PassTurnAction (auto-handled by loop) or Quit.
            # Just show Pass and Quit options.
            tk.Label(
                self.widgets["action_buttons_frame"],
                text="Farkle! Only options are Pass or Quit.",
                fg="red",
            ).pack(side=tk.LEFT)
            tk.Button(
                self.widgets["action_buttons_frame"],
                text="Pass Turn",
                command=self.handle_pass,
            ).pack(side=tk.LEFT, padx=5)
            # Quit button is already globally available

        elif turn_state.has_rolled:
            # If not a Farkle, player can potentially Continue, Bank, or Pass.
            # Continue and Bank require selecting a pattern first.

            # Pass button (forfeit)
            tk.Button(
                self.widgets["action_buttons_frame"],
                text="Pass Turn (Forfeit Score)",
                command=self.handle_pass,
            ).pack(side=tk.LEFT, padx=5)

            # Continue and Bank buttons depend on selected dice forming a valid pattern
            # These buttons will be dynamically configured and enabled by check_selected_pattern
            self.widgets["continue_button"] = tk.Button(
                self.widgets["action_buttons_frame"],
                text="Continue (Select Dice)",
                command=self.handle_continue,
                state=tk.DISABLED,
            )
            self.widgets["continue_button"].pack(side=tk.LEFT, padx=5)

            self.widgets["bank_button"] = tk.Button(
                self.widgets["action_buttons_frame"],
                text="Bank (Select Dice)",
                command=self.handle_bank,
                state=tk.DISABLED,
            )
            self.widgets["bank_button"].pack(side=tk.LEFT, padx=5)

        # Check if game is over
        if state.winner is not None:
            self.widgets["game_status_label"].config(
                text=f"GAME OVER! Player {state.winner + 1} wins!", fg="green"
            )
            self.widgets["message_label"].config(text="")  # Clear any Farkle message
            self.disable_game_interaction()  # Disable buttons, dice clicks

        elif state.is_quit:
            self.widgets["game_status_label"].config(text="Game Quit.", fg="orange")
            self.widgets["message_label"].config(text="")
            self.disable_game_interaction()

        # Ensure the check_selected_pattern is called after dice are updated
        # It will disable Continue/Bank buttons if no dice are rolled or no valid pattern is selected.
        self.check_selected_pattern()

    def on_die_click(self, index: int):
        """Handles a click on a die label."""
        if not self.state.turn_state.has_rolled:  # FARKLE MISSING
            # Cannot select dice if not currently showing a rolled hand or if it was a farkle
            return

        if index in self.selected_dice_indices:
            # Deselect the die
            self.selected_dice_indices.remove(index)
            self.widgets["dice_labels"][index].config(relief=tk.RAISED, bg="white")
        else:
            # Select the die
            self.selected_dice_indices.append(index)
            self.widgets["dice_labels"][index].config(relief=tk.SUNKEN, bg="lightblue")

        # After changing selection, check if it forms a valid pattern and update buttons
        self.check_selected_pattern()

    def check_selected_pattern(self):
        """Checks if the currently selected dice form a scoring pattern and updates Continue/Bank buttons."""
        state = self.state
        turn_state = state.turn_state

        # Reset button states and text
        continue_btn = self.widgets.get("continue_button")
        bank_btn = self.widgets.get("bank_button")

        if continue_btn:
            continue_btn.config(state=tk.DISABLED, text="Continue (Select Dice)")
        if bank_btn:
            bank_btn.config(state=tk.DISABLED, text="Bank (Select Dice)")

        if not self.selected_dice_indices:
            self.widgets["message_label"].config(text="")
            return  # No dice selected, no pattern possible

        # Get the values of the selected dice, sorted
        selected_values = sorted(
            state.turn_state.rolled_dice[i] for i in self.selected_dice_indices
        )
        selected_pattern = tuple(selected_values)

        # Find if this pattern is a known scoring pattern
        score = self.engine.pattern_scores.get(selected_pattern)

        if score is not None:
            # Valid scoring pattern selected
            potential_turn_score = turn_state.score + score
            remaining_dice_count = len(turn_state.rolled_dice) - len(selected_pattern)
            next_roll_count = (
                state.parameters.max_dice_count
                if remaining_dice_count == 0
                else remaining_dice_count
            )

            if continue_btn:
                continue_btn.config(
                    state=tk.NORMAL,
                    text=f"Continue ({potential_turn_score}, roll {next_roll_count} dice) with {list(selected_pattern)} (+{score})",
                )
            if bank_btn:
                bank_btn.config(
                    state=tk.NORMAL,
                    text=f"Bank Total ({potential_turn_score}) with {list(selected_pattern)} (+{score})",
                )
            self.widgets["message_label"].config(
                text=f"Selected pattern {list(selected_pattern)} scores {score}.",
                fg="blue",
            )
        else:
            # Not a valid scoring pattern
            self.widgets["message_label"].config(
                text="Selected dice do not form a valid scoring pattern.", fg="orange"
            )

    def handle_action(self, action: Action):
        """Applies an action using the engine and updates the display."""
        if self.state.winner is not None or self.state.is_quit:
            # Game is over, ignore actions
            return

        try:
            # If Continue or Bank, get the pattern from selected dice
            if isinstance(action, (ContinueAction, BankAction)):
                selected_values = sorted(
                    self.state.turn_state.rolled_dice[i]
                    for i in self.selected_dice_indices
                )
                pattern = tuple(selected_values)
                # Create the correct action instance with the pattern
                action_to_apply: Action = (
                    ContinueAction(pattern)
                    if isinstance(action, ContinueAction)
                    else BankAction(pattern)
                )
            else:
                # Pass or Quit action doesn't need a pattern
                action_to_apply = action

            # Apply the action
            new_state = self.engine.apply_action(self.state, action_to_apply)
            self.state = new_state

            # Clear selected dice and update display *before* handling next turn's roll if needed
            self.selected_dice_indices = []
            self.update_display()

        except ValueError as e:
            messagebox.showerror("Game Error", f"Error applying action: {e}")
            self.update_display()  # Re-display current state on error

    # --- Action Handlers (called by buttons) ---

    def handle_continue(self):
        """Handler for the Continue button."""
        # The pattern is derived from selected_dice_indices in handle_action
        if self.selected_dice_indices:  # Ensure some dice are selected
            self.handle_action(
                ContinueAction(tuple())
            )  # Pattern tuple is placeholder, will be derived
        else:
            messagebox.showwarning(
                "Selection Required", "Please select dice to continue."
            )

    def handle_bank(self):
        """Handler for the Bank button."""
        # The pattern is derived from selected_dice_indices in handle_action
        if self.selected_dice_indices:  # Ensure some dice are selected
            self.handle_action(
                BankAction(tuple())
            )  # Pattern tuple is placeholder, will be derived
        else:
            messagebox.showwarning("Selection Required", "Please select dice to bank.")

    def handle_pass(self):
        """Handler for the Pass Turn button."""
        # Optional: Ask for confirmation if turn score is > 0
        if len(self.engine.actions(self.state)) > 2:
            if not messagebox.askyesno(
                "Confirm Pass",
                f"Passing will forfeit your current turn score of {self.state.turn_state.score}. Are you sure?",
            ):
                return  # Player cancelled
        self.handle_action(PassTurnAction())

    def handle_quit(self):
        """Handler for the Quit button."""
        if messagebox.askyesno("Confirm Quit", "Are you sure you want to quit?"):
            self.handle_action(QuitAction())
            self.master.destroy()  # Close the window after quitting

    def start_game(self):
        """Initializes the first turn of the game."""
        print("Farkle Game Started!")  # Keep console message for debug
        self.state = self.state.roll_dice()  # Roll for the first player's turn
        self.update_display()
        print(f"Player {self.state.current_player + 1}'s Turn Starts.")  # Debug print

    def disable_game_interaction(self):
        """Disables buttons and dice interaction when game is over."""
        for widget in self.widgets["action_buttons_frame"].winfo_children():
            widget.config(state=tk.DISABLED)
        for dice_label in self.widgets["dice_labels"]:
            dice_label.unbind("<Button-1>")  # Remove click binding
            dice_label.config(
                relief=tk.FLAT, bg="lightgray"
            )  # Visually indicate disabled
