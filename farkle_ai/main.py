"""Starts the farkle game loop and loads GUI."""

import tkinter as tk
from game import GameState, Parameters, FarkleEngine, FarkleGUI

if __name__ == "__main__":
    # --- Game Setup ---
    # Could prompt for players/score here later if desired
    params = Parameters(player_count=2, score_to_win=5000)
    initial_state = GameState(parameters=params)
    engine = FarkleEngine()

    # --- GUI Setup ---
    root = tk.Tk()
    gui = FarkleGUI(root, initial_state, engine)

    # --- Start GUI Main Loop ---
    root.mainloop()

    print("Game Over.")
