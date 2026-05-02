import argparse

import torch
import numpy as np
from tianshou.data import Batch

import pennywise
import tinygame

import main_pennywise

from pennywise_smallestcoin import PennywiseSmallestCoinPolicy
from pennywise_bestvalue import PennywiseBestValuePolicy
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy

import pygame
import sys


def do_ai(policy, observation):
    # Handle PyGame quit events during AI turns so the window doesn't freeze
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


    # Ask the neural network for the best move
    mask = [i == 1 for i in observation["action_mask"]]

    batch = Batch(
        obs=Batch({
            "obs": np.expand_dims(observation["observation"], axis=0),
            "mask": np.expand_dims(mask, axis=0)
        }),
        info=info
    )

    result = policy(batch)
    return result.act[0]


if __name__ == "__main__":
    args = main_pennywise.get_args()

    # 1. Initialize environment and reset
    env = pennywise.env(clean_start=True, render_mode="human")
    env.reset()

    policy, optim, agents = main_pennywise.get_agents(
        PettingZooEnv(env), args, train=False,
    )

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            env.step(None)
            continue

        # Draw the screen for the current turn
        env.render()

        mask = observation["action_mask"]

        action_chosen = None

        while action_chosen is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # Listen for Mouse Clicks
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = event.pos

                    # Check which button was clicked
                    for action_id, rect in enumerate(env.action_rects):
                        if rect.collidepoint(mouse_pos):
                            # Ensure the clicked action is actually legal
                            if action_id == 4:
                                action_chosen = do_ai(policy.policies[agent], observation)
                            elif mask[action_id] == 1:
                                action_chosen = action_id
                            else:
                                print("Invalid move! You don't have that coin.")

        action = action_chosen

        # Execute the action and move to the next turn
        env.step(action)

    print("Game Over!")
    pygame.quit()
