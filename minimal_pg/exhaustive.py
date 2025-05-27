# TODO: Clean and Integrate with benchmark.py

from minimal_pg.base import OfflineAgent
from minimal_pg.env import Env
import torch
import random
import numpy as np
import itertools
from tqdm import tqdm

class ExhaustiveAgent(OfflineAgent):
    def __init__(self, seed: int = 42):
        # Set the random seed
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
    
    def offline_update(self, env: Env):
        all_policy = list(itertools.product(env.action_set, repeat=len(env.state_set)))
        
        best_policy, best_reward = None, -float('inf')
        for policy in tqdm(all_policy, desc='Exhaustive Search', total=len(all_policy)):
            reward = self.simulate(env, policy)

            if reward > best_reward:
                best_policy, best_reward = policy, reward

        import yaml
        with open('best_policy.yaml', 'a') as f:
            yaml.dump({f"{env.total_servers}-{env.priorities}-{env.free_prob}": {
                'avg_reward': best_reward,
                'policy': best_policy
            }}, f)
            
    def simulate(self, env: Env, policy: list[int], episodes: int = 10, t_max: int = 10_000) -> float:
        total_reward = 0
        for _ in range(episodes):
            state = env.reset()
            for step in range(t_max):
                state_index = env.state_set.index(state)
                action = policy[state_index]
                next_state, reward, _ = env.step(action)
                state = next_state
                total_reward += reward
        average_reward = total_reward / (episodes * t_max)
        return average_reward