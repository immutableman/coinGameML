import numpy as np
from tianshou.policy import BasePolicy
from tianshou.data import Batch

_action_value = [1, 5, 10, 25]

def get_value(action, pot):
    value = _action_value[action]
    for i in [2, 1, 0]:
        cval = _action_value[i]
        remaining = pot[i]
        while value > cval and remaining > 0:
            value -= cval
            remaining -= 1
    return value / _action_value[action]

class PennywiseBestValuePolicy(BasePolicy):
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
            best_action = legal_actions[0]
            best_val = 1
            for action in legal_actions:
                cval = get_value(action, obs)
                if cval < best_val:
                    best_action = action
                    best_val = cval

            chosen_action = best_action
            actions.append(chosen_action)

        # Return a Tianshou Batch containing the chosen actions
        return Batch(act=np.array(actions), state=state)

    def learn(self, batch, **kwargs):
        """Rule-based bots don't update weights, so this just returns an empty dict."""
        return {}