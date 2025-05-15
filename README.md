# AI Agents for the Farkle Dice Game

This repository contains the implementation of the dice game Farkle, a standardized multi-agent environment using the PettingZoo library, and various Artificial Intelligence agents developed to play the game. The project focuses on designing, implementing, and evaluating different AI strategies, particularly training an agent using Reinforcement Learning (RL) within a carefully designed observation and action space.

## Project Overview

The core goal of this project was to explore different AI approaches for a game with uncertainty like Farkle and to provide a framework for comparing their performance. Key aspects include:

*   Implementing the complete rules of Farkle.
*   Creating a PettingZoo-compatible environment for multi-agent interaction and RL training.
*   Designing a simplified action space for RL agents to handle the game's combinatoric complexity.
*   Defining a suitable observation space and reward function for RL.
*   Implementing agents based on different strategies (Naive, Heuristic, RL).
*   Enabling evaluation through AI vs AI matches and providing modes for Human vs AI and Human vs Human play.

## Farkle Game Rules (Brief Summary)

Farkle is a dice game for 2+ players aiming to reach a target score (e.g., 5000). Players take turns rolling six dice. After a roll, they must set aside at least one scoring die or combination (1s, 5s, triples, straights, etc.). Remaining dice can be re-rolled ("Continue") or the turn score can be added to the total score ("Bank"). If a roll yields no scoring dice, it's a "Farkle", and the player loses all points for that turn. Using all six dice to score is "Hot Dice", allowing the player to re-roll all six.

## Agent Types Implemented

The project includes three main types of agents:

1.  **Naive Agent:** Follows simple, fixed rules. It always selects the highest immediate scoring combination available and continues rolling if 3 or more dice remain, banking otherwise.
2.  **Heuristic Agent:** A more sophisticated agent that attempts to calculate the expected value of rolling the remaining dice and compares it against the value of banking the current turn score to make decisions.
3.  **Reinforcement Learning (RL) Agent:** Trained using the PPO algorithm. This agent learns its strategy through self-play in the PettingZoo environment, aiming to maximize its final game win rate based on sparse win/loss rewards.

## Technical Design Highlights

*   **Game Implementation:** Core Farkle game logic is implemented in Python, defining game state, actions, and rule application.
*   **PettingZoo Environment:** The game is wrapped as a [PettingZoo AEC (Asyncronous Environment for Contemporaneous) environment](https://pettingzoo.farama.org/api/aec/). This provides a standard API (`step`, `reset`, `observe`, `rewards`, `terminations`, `truncations`, `infos`) for multi-agent interaction and RL training frameworks like Ray/RLlib.
*   **Action Space Reduction (for RL):** The standard action space (choose *any* scorable combination) is complex. For RL training, it was reduced to a discrete set of actions representing the *intention* to keep the *best* scorable pattern of a specific length (`Continue_N`, `Bank_N`, where N is the number of dice). A `Pass_Turn` action is also included.
*   **Observation Space (for RL):** A fixed-size vector providing relevant game information to the agent, including:
    *   Your remaining score to win.
    *   The opponent's remaining score to win (maximum opponent score in multi-player).
    *   Your current turn score.
    *   The number of dice in the current roll.
    *   The score of the best scorable pattern for each possible length (1 to 6 dice) in the current roll (0 if none exists).
*   **Reward Function (for RL):** A sparse reward signal is used: +1 for winning the game, -1 for losing, and 0 for all intermediate steps.

## Game Modes

The project supports the following playable modes:

*   **Player vs Player (Human vs Human):** Two human players compete.
*   **Player vs AI (Human vs AI):** A human player competes against one of the implemented AI agents.
*   **AI vs AI:** Two AI agents compete, primarily used for evaluating performance metrics like win rate over multiple games.

## Getting Started

To set up and run the project:

1.  **Clone the repository:**
    ```bash
    git clone [Repository URL]
    cd [Repository Directory]
    ```
2.  **Install Dependencies:**
    Ensure you have Python (3.8+) installed. Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    # or manually:
    # pip install pygame pettingzoo gymnasium numpy torch ray[rllib]
    ```
    *(Note: `ray[rllib]` requires specific system dependencies like build tools.)*
3.  **Prepare RL Checkpoint (if using NeuralNetworkAgent):** If you intend to use the `NeuralNetworkAgent` agent, you need a pre-trained Ray/RLlib checkpoint. Place the checkpoint directory in an accessible location and update the `CHECKPOINT_DIR`.
4.  **Run the Game:**
    You can typically run the `main.py` script and select game settings in a terminal interface. The game will run in a pygame GUI.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. *(Note: You should create a LICENSE file with the MIT license text).*