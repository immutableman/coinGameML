import numpy as np
from tianshou.policy import BasePolicy
from tianshou.data import Batch


class PennywiseSmallestCoinPolicy(BasePolicy):
    def __init__(self):
        super().__init__()
        # No neural network, optimizer, or hyperparameters needed!

    def forward(self, batch, state=None, **kwargs):
        """
        Takes the observation batch and returns a batch of actions.
        """
        # Tianshou automatically batches observations.
        # batch.obs contains the dictionary from your PettingZoo environment.
        masks = batch.obs["mask"]
        observations = batch.obs["obs"]

        actions = []

        # Loop through each agent in the current batch
        for i in range(len(masks)):
            mask = masks[i]
            obs = observations[i]

            # 1. Identify all mathematically legal moves
            legal_actions = np.where(mask == 1)[0]


            # 2. Implement your procedural logic
            # EXAMPLE RULE: "Always pick the highest action ID available"
            # In your game, you would replace this with logic that analyzes 'obs'
            # to execute a specific strategy (e.g., "Always play a Penny if I have one").

            chosen_action = legal_actions[0]
            actions.append(chosen_action)

        # Return a Tianshou Batch containing the chosen actions
        return Batch(act=np.array(actions), state=state)

    def learn(self, batch, **kwargs):
        """Rule-based bots don't update weights, so this just returns an empty dict."""
        return {}