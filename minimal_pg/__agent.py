from abc import ABC, abstractmethod
import itertools
from typing import Union
import torch
import numpy as np
import random
from tqdm import tqdm
import yaml
import warnings

from minimal_pg.env import Action, State, Env
from minimal_pg.base import OnlineAgent, OfflineAgent
from torch.distributions import Bernoulli

OBS_DIM = 2

class ExactCompatiblePolicyGradientAgent(ActorCriticAgent):
    def __init__(self, config_path: str = "config/agent_actor-critic.yml", actor_critic: ActorCriticModule = None, seed: int = 42):
        # Set the random seed
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if actor_critic is None:
            # Default to LinearModules for actor and critic if none provided
            actor_critic = ActorCriticModule(LinearModule(), LinearModule())
        else:
            warnings.warn(
                "Using custom actor_critic module for CompatibleExactPolicyGradientAgent. Ensure compatibility.",
                UserWarning,
                stacklevel=2
            )
        self.actor_critic = actor_critic

        self.avg_reward = 0.0 # Baseline for REINFORCE-style advantage and part of TD error

        # Load hyperparameters from config
        self.alpha = config['alpha'] # Learning rate for avg_reward update
        self.lr_critic = config['lr_critic'] # Learning rate for critic
        self.lr_actor = config['lr_actor'] # Learning rate for actor

        self.last_action = None # Stores the last action taken
    
    def step(self, state: State, output_probs: bool = False) -> Union[Action, torch.Tensor]:
        # Make sure we're in eval mode for inference
        self.actor_critic.eval()
        
        with torch.no_grad():
            logits = self.actor_critic.actor(state)
            action_dist = Bernoulli(logits=logits)
            sampled_action = action_dist.sample()
            self.last_action = sampled_action # Store the sampled action (as a tensor)
        
        # Switch back to train mode
        self.actor_critic.train()
        return sampled_action.item() if not output_probs else action_dist.probs

    def update(self, state: State, next_state: State, reward: float):        
        if self.last_action is None:
            warnings.warn("CompatibleExactPolicyGradientAgent: Update called before step or last_action not set. Skipping update.", UserWarning, stacklevel=2)
            return

        self.actor_critic.zero_grad() # Zero gradients for both actor and critic models
        
        avg_reward_old = self.avg_reward # Store for consistent use in this step

        # Calculate td error
        value_state = self.actor_critic.critic(state)
        value_next_state_detached = self.actor_critic.critic(next_state).detach()
        td_error_for_critic = reward - avg_reward_old + value_next_state_detached - value_state
        
        # Update avg_reward (baseline)
        self.avg_reward += self.alpha * td_error_for_critic.item() # Use .item() if td_error is a tensor

        # Advantage = R - \hat R (same as ExactPolicyGradientAgent )
        advantage_for_actor = reward - avg_reward_old
        
        # Compute phi = \nabla \log \pi(a|s)
        action_logits = self.actor_critic.actor(state)
        dist = Bernoulli(logits=action_logits)
        log_prob = dist.log_prob(self.last_action) 
        log_prob.backward() 
        
        phi = []
        for param in self.actor_critic.actor_model.parameters():
            phi.append(param.grad.clone().detach())

        # Critic Update w = w + \alpha * (td - <w, phi>) * phi
        with torch.no_grad():
            critic_params = list(self.actor_critic.critic_model.parameters())
            # Here we assume
            # 1. critic.parameters() and actor.parameters() have the same sequence length
            # 2. For each parameter in the sequence, the shape of the parameter is the same
            for i, param_w in enumerate(critic_params):
                phi_component = phi[i]
                dot_product = torch.dot(param_w, phi_component)
                param_w += self.lr_critic * (td_error_for_critic.detach() - dot_product) * phi_component
        
        # Actor Compatible Update (using critic parameters)
        with torch.no_grad():
            actor_model_params = list(self.actor_critic.actor_model.parameters())
            critic_model_params = list(self.actor_critic.critic_model.parameters())
            
            # Assuming a 1-to-1 correspondence in parameters for simplicity, as in original compatible features.
            # This is a strong assumption and depends on actor and critic architectures.
            num_params_to_update = min(len(actor_model_params), len(critic_model_params))
            for i in range(num_params_to_update):
                actor_param = actor_model_params[i]
                critic_param_w = critic_model_params[i]
                if actor_param.shape == critic_param_w.shape:
                     actor_param += self.lr_actor * critic_param_w
                # Else: shapes don't match, skip or error. Compatible features usually assume this match.
        
if __name__ == '__main__':
    # Set seed for reproducibility
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    env = Env(seed=seed_value)
    agent = ExhaustiveAgent(seed=seed_value)
    agent.offline_update(env)