from pettingzoo.utils.env import AECEnv
from gymnasium import spaces
import numpy as np

import os
import pygame

from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector

def get_image(path):
    """Return a pygame image loaded from the given path."""
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    return image

def env(**kwargs):
    env = SimpleEnv(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

SCALE = 64

class SimpleEnv(AECEnv):
    def __init__(
        self, render_mode: str | None = None, screen_height: int | None = 1000
    ):
        super().__init__()
        self.agents = ["player_0", "player_1", "player_2", "player_3"]
        self.possible_agents = self.agents[:]

        # dummy game, just don't give up
        self.TOTAL_ACTIONS = 2

        self.action_spaces = {a: spaces.Discrete(self.TOTAL_ACTIONS) for a in self.agents}

        # The observation space must be a Dictionary containing the mask for MARL
        self.observation_spaces = {
            a: spaces.Dict({
                # 20 integers: 4 for Pot + (4 for each of the 4 players)
                "observation": spaces.MultiDiscrete([16] * 20),
                "action_mask": #spaces.Box(0, 1, shape=(self.TOTAL_ACTIONS,), dtype=np.int8)
                spaces.Discrete(self.TOTAL_ACTIONS)
            }) for a in self.agents
        }

        self.render_mode = render_mode
        self.screen_width = int(1.7 * screen_height)
        self.screen_height = screen_height
        self.screen = None


    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        """Constructs the observation strictly from the current agent's perspective."""
        pot = self.state["pot"]

        # Shift the array so the current 'agent' is always the first inventory block
        agent_idx = self.agents.index(agent)
        ordered_inventories = []
        for i in range(4):
            idx = (agent_idx + i) % 4
            ordered_inventories.extend(self.state["players"][self.agents[idx]])

        # Flatten into a single 20-element array
        obs = np.array(pot + ordered_inventories, dtype=np.int32)

        # Calculate which of the actions are actually legal right now
        mask = self._generate_action_mask(agent)

        return {"observation": obs, "action_mask": mask}

    def _generate_action_mask(self, agent):
        mask = [1,1]
        return mask

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self._out = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.state = {
            "pot": [0, 0, 0, 0],  # Pennies, Nickels, Dimes, Quarters
            "players": {a: [4, 3, 2, 1] for a in self.agents}
        }

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

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

        # self._clear_rewards()

        # 2. Execute action (e.g., move coins between inventory and pot)
        self._execute_action(agent, action)

        # 3. Check for Elimination (Inventory hits 0)
        if not self._out[agent]:
            if sum(self.state["players"][agent]) == 0:
                # self.terminations[agent] = True
                self._out[agent] = True

        # 4. Check for Winner (Last player standing)
        if len(active_agents) <= 1:
            winner = active_agents[0]
            # self.terminations[winner] = True

            # End the game for all agents
            for a in self.agents:
                self.terminations[a] = True
                if a == winner:
                    self.rewards[winner] = 1
                else:
                    self.rewards[a] = -1

            #print(self.rewards)
            self._accumulate_rewards()

        #print("==>", self.state["players"][agent], self.state["pot"])

        # 5. PettingZoo turn progression
        self.agent_selection = self._agent_selector.next()

    def _execute_action(self, agent, action):
        if self.terminations[agent] or self._out[agent]:  # Ignore eliminated players
            return

        inventory = self.state['players'][agent]
        pot = self.state['pot']
        if action == 1:
            pot[0] += inventory[0]
            pot[1] += inventory[1]
            pot[2] += inventory[2]
            pot[3] += inventory[3]
            inventory[0] = 0
            inventory[1] = 0
            inventory[2] = 0
            inventory[3] = 0
        else:
            for i, c in enumerate(inventory):
                if c > 0:
                    pot[i] += 1
                    inventory[i] -= 1
                    break

    def render(self):
        if self.render_mode is None:
            return

        screen_height = self.screen_height
        screen_width = self.screen_width

        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))

            def draw_one(inv, loc):
                for i, c in enumerate(inv):
                    for j in range(c):
                        self.screen.blit(self._coin_img[i], (loc[0] + SCALE//2 * j, loc[1] + SCALE * i))
            draw_one(self.state['pot'], (screen_width//2 - SCALE//2 * 8, screen_height//2 - SCALE*2))
            for i, a in enumerate(self.agents):
                draw_one(self.state['players'][a], (i//2 * (screen_width - SCALE*4), i%2 * (screen_height - SCALE*5)))

            pygame.display.update()
                # self.clock.tick(self.metadata["render_fps"])
        #
        # observation = np.array(pygame.surfarray.pixels3d(self.screen))
        #
        # return (
        #     np.transpose(observation, axes=(1, 0, 2))
        #     if self.render_mode == "rgb_array"
        #     else None
        # )
