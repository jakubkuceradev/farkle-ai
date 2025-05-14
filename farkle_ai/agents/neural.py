"""Support for NeuralNetworkAgent."""

import os
import torch
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

from farkle_ai.environment import ENV_NAME, env as env_creator
from .agent import Agent

CONFIG = (
    PPOConfig()
    .environment(
        env=ENV_NAME,
    )
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=(lambda aid, *args, **kwargs: "shared_policy"),
    )
    .framework("torch")
)

CHECKPOINT_DIR = os.path.abspath(
    "/home/nervyl/Projects/farkle_ai/farkle_ai/checkpoints/FarklePPO"
)

POLICY_ID = "shared_policy"


def petting_zoo_env_creator(env_config):
    """Return PettingZooEnv wrapped farkle environment."""

    env = env_creator(**env_config)
    return PettingZooEnv(env)


class NeuralNetworkAgent(Agent):
    """An agent that uses a trained Ray/RLlib policy (PyTorch backend) to compute actions."""

    def __init__(
        self,
    ):
        """Initializes the agent by loading a trained policy from a Ray/RLlib checkpoint."""

        self.checkpoint_path = CHECKPOINT_DIR

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint path not found: {self.checkpoint_path}"
            )

        register_env(ENV_NAME, petting_zoo_env_creator)

        self.algo = CONFIG.build()
        self.algo.restore(self.checkpoint_path)
        self.module = self.algo.get_module(POLICY_ID)
        if self.module is None:
            raise ValueError(
                f"Could not get policy with ID '{POLICY_ID}' from the restored algorithm."
            )

    def compute_action(self, observation: np.ndarray, mask: np.ndarray) -> int:
        """Computes action using a trained neural network."""

        batched = np.expand_dims(observation, axis=0)
        input_tensor = torch.from_numpy(batched).float()
        forward_input = {"obs": input_tensor}
        action_logits_dict = self.module.forward_inference(forward_input)
        action_logits = action_logits_dict["action_dist_inputs"]
        mask_tensor = torch.from_numpy(mask).bool()
        inf_mask = torch.clamp(torch.log(mask_tensor), min=-float("inf"))
        masked_logits = action_logits + inf_mask
        action = torch.argmax(masked_logits[0]).cpu().item()
        return action
