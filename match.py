import argparse
from pettingzoo import AECEnv
from farkle_ai.agents import (
    Agent,
    BankHighestAgent,
    NaiveAgent,
    RandomAgent,
    NeuralNetworkAgent,
)
from farkle_ai.environment import env as farkle_env, action_mask


def run_match(env: AECEnv, agents: list[Agent], render_mode=None):
    """Run a single match between agents in a pettingzoo environment."""

    agent_mapping = {f"player_{i}": agent for i, agent in enumerate(agents)}

    env.reset()

    if render_mode:
        env.render()

    step_counter = 0

    for agent in env.agent_iter():
        step_counter += 1
        if step_counter > 200:
            print("Max steps reached.")
            break

        observation, reward, termination, truncation, info = env.last()
        mask = action_mask(env.state())

        if termination or truncation:
            env.step(None)
            continue

        model = agent_mapping[agent]
        action = model.compute_action(observation, mask)
        env.step(action)


def run_matches(
    agents: list[Agent], num_matches: int = 500, render_mode=None
) -> dict[str, dict[str, int | float]]:
    """Run multiple matches between agents and return the result."""

    player_count = len(agents)
    agent_mapping = {f"player_{i}": agent for i, agent in enumerate(agents)}
    results = {
        str(agent): {"wins": 0, "losses": 0, "draws": 0, "winrate": 0.0}
        for agent in agents
    }

    print(f"Running {num_matches} matches...")

    env = farkle_env(player_count=player_count, render_mode=render_mode)

    for match in range(num_matches):

        if match % 50 == 0:
            print(f"Running match {match}...")

        if match == num_matches // 2:
            agents[0], agents[-1] = agents[-1], agents[0]

        run_match(env, agents, render_mode)

        winner = env.state().winner

        for i, agent in enumerate(agents):
            agent_name = str(agent)
            if winner is None:
                results[agent_name]["draws"] += 1
            elif winner == i:
                results[agent_name]["wins"] += 1
            else:
                results[agent_name]["losses"] += 1

    for agent in agents:
        results[str(agent)]["winrate"] = results[str(agent)]["wins"] / num_matches

    print("Simulation finished.")

    return results


# --- Example Usage ---


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "agents", metavar="AGENTS", type=str, nargs="+", help="A list of agent names"
    )

    parser.add_argument(
        "--num-matches",
        "-n",
        type=int,
        default=100,
        help="Number of matches to play",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="""Whether to use render_mode = human.""",
    )

    agent_mapping = {
        "naive": NaiveAgent,
        "bank": BankHighestAgent,
        "random": RandomAgent,
        "neural": NeuralNetworkAgent,
    }

    args = parser.parse_args()

    agents = []
    for agent_name in args.agents:
        agent_type = agent_mapping.get(agent_name, None)
        if agent_type is None:
            raise ValueError("{agent_name} is not a valid agent name")
        agents.append(agent_type())

    num_matches = args.num_matches
    render_mode = "human" if args.render else None

    print(run_matches(agents, num_matches, render_mode))
