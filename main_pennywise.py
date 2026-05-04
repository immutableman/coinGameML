import pennywise
import tinygame

from pennywise_smallestcoin import PennywiseSmallestCoinPolicy
from pennywise_bestvalue import PennywiseBestValuePolicy

import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple
import random

import gymnasium
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter



def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1629)
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument("--eps-train", type=float, default=0.5)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gamma", type=float, default=0.995, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--n-step", type=int, default=10000)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    # parser.add_argument(
    #     "--hidden-sizes", type=int, nargs="*", default=[64, 64]
    # )
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128]
    )
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.5,
        help="the expected winning rate: Optimal policy can get 0.7",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action='store_true',
        help="no training, " "watch the play of pre-trained models",
    )
    parser.add_argument(
        "--train-agents", type=int, nargs="*", default=[0, 0, 0, 0],
        help="0==trained, 1==random, 2==smallest, 3==bestvalue, *==custom"
    )
    parser.add_argument(
        "--test-agents", type=int, nargs="*", default=[0, 3, 3, 3],
        help="0==trained, 1==random, *==custom"
    )

    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="the path of agent pth file " "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--league", action='store_true', help="use league for training")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

def get_agents(
    env,
    args: argparse.Namespace = get_args(),
    train: bool = True,
    # agent_learn: Optional[BasePolicy] = None,
    # agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    ).to(args.device)
    if optim is None:
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    base_agent = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
    )

    def choose_policy(type):
        if i == 0:
            return base_agent
        elif i == 1:
            return RandomPolicy()
        elif i == 2:
            return PennywiseSmallestCoinPolicy()
        elif i == 3:
            return PennywiseBestValuePolicy()
        else:
            agent = deepcopy(base_agent)
            fn = os.path.join("pennywise", "checkpoints", "%d-stashed.pth" % i)
            agent.load_state_dict(torch.load(fn))
            return agent

    if args.resume_path:
        fn = os.path.join("pennywise", "checkpoints", args.resume_path)
        base_agent.load_state_dict(torch.load(fn))
    if train:
        agents = []
        for i in args.train_agents:
            agents.append(choose_policy(i))
    else:
        agents = []
        for i in args.test_agents:
            agents.append(choose_policy(i))
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents

def get_env(clean_start=False, render_mode=None):
    return PettingZooEnv(pennywise.env(clean_start=clean_start, render_mode=render_mode))


def train_agent(
    args: argparse.Namespace = get_args(),
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:
    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    env = get_env()

    league_pool = [RandomPolicy(), PennywiseSmallestCoinPolicy()]

    # ======== agent setup =========
    policy, optim, agents = get_agents(
        env, args, train=True, optim=optim  # agent_learn=agent_learn, agent_opponent=agent_opponent,
    )

    # ======== collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    test_policy, _, _ = get_agents(
        env, args, train=False, optim=optim  # agent_learn=agent_learn, agent_opponent=agent_opponent,
    )
    test_collector = Collector(test_policy, test_envs, exploration_noise=True)
    # test_collector = Collector(policy, test_envs, exploration_noise=True)

    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, "pennywise", "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # ======== callback functions used during training =========
    training_state = {"current_epoch": 0}

    def save_best_fn(policy):
        epoch = training_state["current_epoch"]
        agent_policy = policy.policies[agents[0]]
        fn = os.path.join("pennywise", "checkpoints", "gen-%.8d.pth" % epoch)
        torch.save(
            agent_policy.state_dict(), fn
        )

        snapshot = deepcopy(agent_policy)
        snapshot.eval()
        snapshot.set_eps(0.)

        league_pool.append(snapshot)

    def stop_fn(mean_rewards):
        return False
        return mean_rewards >= args.win_rate

    def train_fn(epoch, env_step):
        # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)
        # Decay over the first 80% of training time
        decay_epochs = args.epoch * 0.8
        if epoch <= decay_epochs:
            # Linear decay formula
            epsilon = args.eps_train - (epoch / decay_epochs) * (args.eps_train - 0.1)
        else:
            # Floor out at the minimum epsilon for the rest of training
            epsilon = 0.1
        policy.policies[agents[0]].set_eps(epsilon)

        if args.league:
            # Draft 3 random opponents from the historical pool
            drafted_opponents = random.choices(league_pool, k=3)

            # Inject the new roster into Tianshou's active MultiAgent manager
            # (Assuming your env.agents is a list like ['player_0', 'player_1', 'player_2', 'player_3'])
            for idx, agent_id in enumerate(env.agents):
                if idx > 0:
                    policy.policies[agent_id] = drafted_opponents[idx-1]

    def test_fn(epoch, env_step):
        training_state["current_epoch"] = epoch
        # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
        test_policy.policies[agents[0]].set_eps(args.eps_test)
        # policy.policies[agents[0]].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, 0]
        # return rews[:, args.agent_id - 1]

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    return result, policy.policies[agents[0]]
    # return result, policy.policies[agents[args.agent_id - 1]]


# ======== a test function that tests a pre-trained agent ======
def watch(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
) -> None:
    mode = "human"
    render = args.render
    if args.test_num > 1 or args.render == 0:
        mode = None
        render = 0
    watchc = 1
    if args.test_num >= 1:
        watchc = args.test_num

    def ge():
        return get_env(clean_start=True, render_mode=mode)
    test_envs = DummyVectorEnv([ge for _ in range(watchc)])
    # FORCE RANDOM SEEDS: Give each of the 10 environments a unique, random seed
    random_seeds = [np.random.randint(0, 100000) for _ in range(watchc)]
    test_envs.seed(random_seeds)

    #env = DummyVectorEnv([lambda: get_env(clean_start=True, render_mode=mode)])
    policy, optim, agents = get_agents(
        ge(), args, train=False #, agent_learn=agent_learn, agent_opponent=agent_opponent
    )
    policy.eval()
    # DISABLING EPS FOR WATCHING
    #policy.policies[agents[0]].set_eps(args.eps_test)
    # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
    collector = Collector(policy, test_envs, exploration_noise=True)
    result = collector.collect(n_episode=watchc, render=render)
    rews, lens = result["rews"], result["lens"]
    print(f"P1 reward: {rews[:, 0].mean()}, length: {lens.mean()}")
    print(f"P2 reward: {rews[:, 1].mean()}, length: {lens.mean()}")
    print(f"P3 reward: {rews[:, 2].mean()}, length: {lens.mean()}")
    print(f"P4 reward: {rews[:, 3].mean()}, length: {lens.mean()}")
    # print(f"Final reward: {rews[:, args.agent_id - 1].mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = get_args()
    print('Using ', args.device)
    if args.watch == 0:
        result, agent = train_agent(args)
        watch(args, agent)
    else:
        watch(args)