"""Trains a RL model for the abstract farkle pettingzoo environment"""

import os
import argparse
import torch
import ray
from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from farkle_ai.environment import env as env_creator, ENV_NAME

CHECKPOINT_DIR = os.path.abspath(
    "/home/nervyl/Projects/farkle_ai/farkle_ai/checkpoints/FarklePPO"
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

THREAD_COUNT = 16
SAVE_INTERVAL = 10


def petting_zoo_env_creator(env_config):
    """Return wrapped farkle environment."""

    env = env_creator(**env_config)
    return PettingZooEnv(env)


def main(num_iters: int = 1000, start_new=False):
    """Start training from the latest checkpoint."""

    ray.init()

    register_env(ENV_NAME, petting_zoo_env_creator)

    config = (
        PPOConfig()
        .environment(
            env=ENV_NAME,
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=(lambda aid, *args, **kwargs: "shared_policy"),
        )
        .training(train_batch_size_per_learner=5000, minibatch_size=500)
        .env_runners(num_env_runners=THREAD_COUNT)
        .learners(num_learners=1, num_gpus_per_learner=1)
        .framework("torch")
        .callbacks(
            on_episode_end=(
                lambda episode, **kw: print(f"Episode done. R={episode.get_return()}")
            )
        )
    )

    # Build the Algorithm
    algo = config.build_algo()
    if not start_new:
        try:
            algo.restore(CHECKPOINT_DIR)
            print("Restoration successful.")
        except Exception as e:
            print(f"Direct restore failed: {e}. Attempting from_checkpoint...")
            try:
                algo = algo.from_checkpoint(CHECKPOINT_DIR)
                print("Restoration via from_checkpoint successful.")
            except Exception as e_chkpt:
                print(f"Restoration via from_checkpoint also failed: {e_chkpt}")
                print("Cannot restore checkpoint. Exiting.")
                ray.shutdown()
                return  # Exit if restore fails

        def betas_tensor_to_float_fix(learner):
            try:
                first_opt_param_groups = next(
                    iter(learner._optimizer_parameters.keys())
                ).param_groups
                param_group = first_opt_param_groups[0]

                if (
                    "betas" in param_group
                    and isinstance(param_group["betas"], tuple)
                    and len(param_group["betas"]) > 0
                    and isinstance(param_group["betas"][0], torch.Tensor)
                ):
                    if not param_group.get("capturable", False):
                        print("Applying betas tensor to float workaround...")
                        param_group["betas"] = tuple(
                            beta.item() for beta in param_group["betas"]
                        )
                        print("Betas converted successfully.")
                    else:
                        print(
                            "Betas are Tensors but capturable is True. Workaround skipped."
                        )
                else:
                    print(
                        f"Betas not found or not Tensors in first param group, or already floats. Type: {type(param_group.get('betas', None))}. capturable: {param_group.get('capturable', 'Not Found')}"
                    )

            except AttributeError as e:
                print(
                    f"AttributeError accessing optimizer parameters in workaround: {e}"
                )

            except StopIteration:
                print(
                    "StopIteration: No optimizers found in learner._optimizer_parameters"
                )

            except Exception as e:
                print(f"An unexpected error occurred in betas_tensor_to_float_fix: {e}")

        print("Applying betas tensor to float workaround across learners...")
        algo.learner_group.foreach_learner(betas_tensor_to_float_fix)
        print("Finished applying workaround.")

    for i in range(num_iters):
        print(f"\nTraining Iteration {i+1}/{num_iters}")
        algo.train()

        if i % SAVE_INTERVAL == 0:
            checkpoint_path = algo.save(CHECKPOINT_DIR)
            print(f"Checkpoint saved at: {checkpoint_path.checkpoint.path}")

    print("\nTraining complete.")
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="The number of training iterations to run.",
    )

    parser.add_argument(
        "--new",
        type=bool,
        default=False,
        help="Create a new checkpoint and overwrite the old one.",
    )

    args = parser.parse_args()

    main(args.iters, args.new)
