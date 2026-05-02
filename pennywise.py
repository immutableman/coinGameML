from pettingzoo.utils.env import AECEnv
from gymnasium import spaces
import numpy as np

import os
import pygame

from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector
import random

width = [0, 1, 1, 0]
height = [0, 0, 1, 1]

def get_image(path):
    """Return a pygame image loaded from the given path."""
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    return image

def env(**kwargs):
    env = PennywiseEnv(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

SCALE = 64

class PennywiseEnv(AECEnv):
    def __init__(
        self, clean_start: bool = False, render_mode: str | None = None, screen_height: int | None = 1000
    ):
        super().__init__()
        self.agents = ["player_0", "player_1", "player_2", "player_3"]
        self.possible_agents = self.agents[:]

        # play penny, nickel, dime, quarter. assume you must take most valuable change.
        self.TOTAL_ACTIONS = 4
        self._action_value = [1, 5, 10, 25]

        self.action_spaces = {a: spaces.Discrete(self.TOTAL_ACTIONS) for a in self.agents}

        # The observation space must be a Dictionary containing the mask for MARL
        self.observation_spaces = {
            a: spaces.Dict({
                # 20 integers: 4+1 for Pot + (4+1 for each of the 4 players)
                "observation": spaces.MultiDiscrete([120] * 25),
                "action_mask": #spaces.Box(0, 1, shape=(self.TOTAL_ACTIONS,), dtype=np.int8)
                spaces.Discrete(4)
            }) for a in self.agents
        }
        self.clean_start = clean_start

        self.render_mode = render_mode
        self.screen_width = int(1.7 * screen_height)
        self.screen_height = screen_height
        self.screen = None

        # Define clickable zones for the 4 actions: [x, y, width, height]
        self.action_rects = [
            pygame.Rect(560, 64, 64, 64),  # Action 0: Penny
            pygame.Rect(660, 64, 64, 64),  # Action 1: Nickel
            pygame.Rect(760, 64, 64, 64),  # Action 2: Dime
            pygame.Rect(860, 64, 64, 64),  # Action 3: Quarter
            pygame.Rect(960, 64, 64, 64),  # Action 4: AI choice
        ]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def value(self, inventory):
        v = 0
        for i,c in enumerate(inventory):
            v += self._action_value[i] * c
        return v

    def observe(self, agent):
        """Constructs the observation strictly from the current agent's perspective."""
        pot = self.state["pot"]

        # Shift the array so the current 'agent' is always the first inventory block
        agent_idx = self.agents.index(agent)
        ordered_inventories = []
        for i in range(4):
            idx = (agent_idx + i) % 4
            inv = self.state["players"][self.agents[idx]]
            ordered_inventories.extend(inv)
            ordered_inventories.append(self.value(inv))

        # Flatten into a single 20-element array
        obs = np.array(pot + [self.value(pot)] + ordered_inventories, dtype=np.int32)

        # Calculate which of the actions are actually legal right now
        mask = self._generate_action_mask(agent)

        return {"observation": obs, "action_mask": mask}

    def _generate_action_mask(self, agent):
        """Returns an array of 0s and 1s indicating legal moves."""
        #mask = np.zeros(self.TOTAL_ACTIONS, dtype=np.int8)
        if self.terminations[agent] or self._out[agent]:  # Ignore eliminated players
            return np.ones(self.TOTAL_ACTIONS, dtype=np.int8)

        inventory = self.state["players"][agent]
        # Logic to check player inventory and pot contents goes here
        # If Action ID 5 is legal, mask[5] = 1
        mask = [0 if c == 0 else 1 for c in inventory]
        return mask

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self._out = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        if self.clean_start:
            self.state = {
                "pot": [0, 0, 0, 0],
                "players": {a: [4, 3, 2, 1] for a in self.agents},
                "scores": {a: 0 for a in self.agents}
            }
        else:
            pot = random.choices([[0,0,0,0],[1,0,0,0],[0,1,0,0],[2,0,0,0],[0,1,0,0],[1,1,0,0],[3,0,0,0],[0,1,0,0],[1,1,0,0],[0,0,1,0]],
                                   [1,      0.5,    0.5,    0.25,   0.5,    0.25,   0.25,   0.25,   0.25,   0.25])
            self.state = {
                "pot": pot[0],
                "players": {a: [random.randint(3,4),
                                random.randint(2,3),
                                2,
                                1] for a in self.agents},
                "scores": {a: 0 for a in self.agents}
            }

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # self._update_score()

        if self.render_mode is not None and self.screen is None:
            pygame.init()

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Pennywise")

            self._coin_img = [get_image(os.path.join("img", "penny.png")),
                    get_image(os.path.join("img", "nickel.png")),
                    get_image(os.path.join("img", "dime.png")),
                    get_image(os.path.join("img", "quarter.png"))]
            self._coin_img = [pygame.transform.scale(i, (SCALE, SCALE)) for i in self._coin_img]

    def step(self, action):
        agent = self.agent_selection
        active_agents = [a for a in self.agents if not self._out[a]]

        #print(agent, action, self.state["players"][agent], self.state["pot"], self._out[agent], len(active_agents))

        # if len(active_agents) <= 1:
        #     self._was_dead_step(action)
        #     return

        # 1. Handle already-eliminated agents (PettingZoo requirement)
        # if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
        #     self._was_dead_step(action)
        #     return

        self._clear_rewards()

        # 2. Execute action (e.g., move coins between inventory and pot)
        value = self._execute_action(agent, action)
        #self._update_score()

        # 3. Check for Elimination (Inventory hits 0)
        if not self._out[agent]:
            if sum(self.state["players"][agent]) == 0:
                # self.terminations[agent] = True
                self._out[agent] = True

        # 4. Check for Winner (Last player standing)
        if len(active_agents) <= 1:
            winner = active_agents[0]

            # End the game for all agents
            for a in self.agents:
                self.terminations[a] = True
                if a == winner:
                    self.rewards[winner] = 1
                else:
                    self.rewards[a] = -1

            #print(self.rewards)
            self._accumulate_rewards()
        else:
            for a in self.agents:
                if a == agent:
                    self.rewards[a] = value / 100.0
                else:
                    self.rewards[a] = 0
            # print(agent, action, value, self.rewards)
            self._accumulate_rewards()

        #print("==>", self.state["players"][agent], self.state["pot"])

        # 5. PettingZoo turn progression
        self.agent_selection = self._agent_selector.next()

    def _update_score(self):
        for agent in self.state["scores"]:
            score = 0
            for a, inv in self.state["players"].items():
                s = 1 if a == agent else -1
                for i, c in enumerate(inv):
                    score += s * c * self._action_value[i]

            self.rewards[agent] = score - self.state["scores"][agent]
            self.state["scores"][agent] = score
        self._accumulate_rewards()
        # print(self.rewards)
        # print(self.state["players"])
        # print(self.state["scores"])

    def _execute_action(self, agent, action):
        if self.terminations[agent] or self._out[agent]:  # Ignore eliminated players
            return -1

        inventory = self.state['players'][agent]
        pot = self.state['pot']

        # 1. Remove the picked coin
        inventory[action] -= 1

        # 2. Claim any possible coins
        spent = value = self._action_value[action]
        got = 0
        for i in [2, 1, 0]:
            cval = self._action_value[i]
            while value > cval and pot[i] > 0:
                value -= cval
                pot[i] -= 1
                inventory[i] += 1
                got += 1

        # 3. Put the picked coin into pot
        pot[action] += 1
        # return -value
        return value/spent

    def render(self):
        if self.render_mode is None:
            return

        screen_height = self.screen_height
        screen_width = self.screen_width
        agent = self.agent_selection

        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))

            def draw_one(inv, loc, curr):
                if curr:
                    pygame.draw.rect(self.screen, (100, 100, 100), (loc[0], loc[1], SCALE*8, SCALE*4))
                for i, c in enumerate(inv):
                    for j in range(c):
                        self.screen.blit(self._coin_img[i], (loc[0] + SCALE//2 * j, loc[1] + SCALE * i))
            draw_one(self.state['pot'], (screen_width//2 - SCALE//2 * 8, screen_height//2 - SCALE*2), False)
            for i, a in enumerate(self.agents):
                draw_one(self.state['players'][a], (width[i%4] * (screen_width - SCALE*8),
                                                    height[i%4] * (screen_height - SCALE*5)), agent == a)

            # 3. Draw the Action Buttons (Only highlight valid ones for the current player)
            mask = self._generate_action_mask(agent)

            for i, rect in enumerate(self.action_rects):
                if i == 4:
                    color = (200, 200, 0)
                else:
                    color = (0, 200, 0) if mask[i] == 1 else (100, 100, 100)  # Green if valid, Grey if invalid
                pygame.draw.rect(self.screen, color, rect)
                # self.screen.blit(self._coin_img[i], (rect.x + 10, rect.y + 15))

                # self.clock.tick(self.metadata["render_fps"])

            pygame.display.flip()

        #
        # observation = np.array(pygame.surfarray.pixels3d(self.screen))
        #
        # return (
        #     np.transpose(observation, axes=(1, 0, 2))
        #     if self.render_mode == "rgb_array"
        #     else None
        # )
