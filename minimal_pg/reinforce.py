# TODO: Clean and Integrate with benchmark.py

from minimal_pg.__agent import OnlineAgent
from minimal_pg.env import Action, State
import torch
import random
import numpy as np
import yaml
import warnings

class ExactPolicyGradientAgent(OnlineAgent):
    def __init__(self, config_path: str = "config/agent_exact-grad.yml", actor: torch.nn.Module = None, seed: int = 42):
        # Set the random seed
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.actor = actor or LinearModule()
        self.avg_reward = 0.0 # Initialize average reward (rho estimate)

        # Load hyperparameters from config
        self.alpha = config['alpha'] # Learning rate for avg_reward update
        self.lr_actor = config['lr_actor'] # Learning rate for actor

        self.last_action = None # Stores the last action taken

    def step(self, state: State, output_probs: bool = False) -> Union[Action, torch.Tensor]:
        # Make sure we're in eval mode for inference
        self.actor.eval()
        
        with torch.no_grad():
            logits = self.actor(state)

            action_dist = Bernoulli(logits=logits)
            sampled_action = action_dist.sample()
            self.last_action = int(sampled_action)
        
        self.actor.train()
        return sampled_action.item() if not output_probs else action_dist.probs

    def update(self, state: State, next_state: State, reward: float):
        self.actor.train()
        self.actor.zero_grad()
                
        # Calculate Advantage = R - \hat R and update \hat R
        advantage = reward - self.avg_reward
        self.avg_reward += self.alpha * advantage

        # Keep avg_reward as a tensor, convert advantage to tensor after
        advantage = torch.tensor(advantage)
        
        logits = self.actor(state)
        dist = Bernoulli(logits=logits)
        action_tensor = torch.tensor(self.last_action, dtype=logits.dtype) 
        log_prob = dist.log_prob(action_tensor)
        
        # REINFORCE objective: J = E[ln pi(a|s) * A]
        actor_objective = log_prob * advantage
        actor_objective.backward()
        with torch.no_grad(): # Optimizer step
            for param in self.actor.parameters():
                if param.grad is not None:
                    param += self.lr_actor * param.grad