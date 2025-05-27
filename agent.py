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
from torch.distributions import Bernoulli

OBS_DIM = 2

class MLPModule(torch.nn.Module):
    def __init__(self, hidden_dim: int = 4):
        super().__init__()
        self.fc1 = torch.nn.Linear(OBS_DIM, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, state: State) -> torch.Tensor:
        state = torch.tensor(state, dtype=torch.float32) # (OBS_DIM,)
        x = self.fc1(state)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

class LinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize t with mean 0 and std 0.1 to get logits around 0 
        # (which gives Bernoulli probabilities around 0.5)
        self.t = torch.nn.Parameter(torch.zeros(OBS_DIM, dtype=torch.float32))

    def forward(self, state: State) -> torch.Tensor:
        state = torch.tensor(state, dtype=torch.float32) # (OBS_DIM,)
        return torch.dot(self.t, state) # (1,)

class ActorCriticModule(torch.nn.Module):
    """
        Assume action dim = state dim
    """
    def __init__(self, actor: torch.nn.Module = None, critic: torch.nn.Module = None):
        super().__init__()
        self.actor_model = actor
        self.critic_model = critic
    
    def actor(self, state: State) -> torch.Tensor:
        return self.actor_model(state)

    def critic(self, state: State) -> torch.Tensor:
        return self.critic_model(state)

class OnlineAgent(ABC):
    @abstractmethod
    def step(self, state: State) -> Action:
        raise NotImplementedError
    @abstractmethod
    def update(self, state: State, next_state: State, reward: float):
        raise NotImplementedError

class OfflineAgent(ABC):
    @abstractmethod
    def offline_update(self, env: Env):
        raise NotImplementedError

class ActorCriticAgent(OnlineAgent):
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
            actor_critic = ActorCriticModule(LinearModule(), LinearModule())
        self.actor_critic = actor_critic

        self.avg_reward = 0

        # Load hyperparameters from config
        self.alpha = config['alpha']
        self.lr_critic = config['lr_critic']
        self.lr_actor = config['lr_actor']
        self.grad_clip_norm = config['grad_clip_norm']
        self.lambda_v = config['lambda_v']
        self.lambda_entropy = config['lambda_entropy']

        self.last_action = None
        self.debug = True
    
    def step(self, state: State, output_probs: bool = False) -> Union[Action, torch.Tensor]:
        # Make sure we're in eval mode for inference
        self.actor_critic.eval()
        
        with torch.no_grad():
            logits = self.actor_critic.actor(state)

            # print('logits', logits) if self.debug else None
            action_dist = Bernoulli(logits=logits)
            sampled_action = action_dist.sample()
            self.last_action = sampled_action
        
        # Switch back to train mode
        self.actor_critic.train()
        return sampled_action.item() if not output_probs else action_dist.probs

    def update(self, state: State, next_state: State, reward: float):        
        self.actor_critic.zero_grad()
        
        # Compute td
        with torch.no_grad():
            value_state = self.actor_critic.critic(state)
            value_next_state_detached = self.actor_critic.critic(next_state).detach()
            td = reward - self.avg_reward + value_next_state_detached - value_state
            # Update average reward
            self.avg_reward += self.alpha * td
        
        # Update critic
        value_state = self.actor_critic.critic(state)
        # Add regularization term
        critic_reg = self.lambda_v * self._compute_l2_reg(self.actor_critic.critic_model)
        obj_critic = value_state - critic_reg
        obj_critic.backward()
        
        # Clip gradients to prevent explosion or vanishing
        torch.nn.utils.clip_grad_norm_(self.actor_critic.critic_model.parameters(), max_norm=self.grad_clip_norm)
        
        with torch.no_grad(): # Optimizer step
            for param in self.actor_critic.critic_model.parameters():
                param += self.lr_critic * td * param.grad
        
        # Zero gradients before actor update
        self.actor_critic.zero_grad()
        
        # Actor update
        action_logits = self.actor_critic.actor(state)
        dist = Bernoulli(logits=action_logits)
        log_prob = dist.log_prob(self.last_action)
        obj_actor = log_prob + self.lambda_entropy * dist.entropy()
        obj_actor.backward()
        
        # Clip gradients to prevent explosion or vanishing
        torch.nn.utils.clip_grad_norm_(self.actor_critic.actor_model.parameters(), max_norm=self.grad_clip_norm)
        
        with torch.no_grad(): # Optimizer step
            for param in self.actor_critic.actor_model.parameters():
                param += self.lr_actor * td * param.grad

    def _compute_average_param_norm(self, model: torch.nn.Module) -> torch.Tensor:
        parameters = torch.tensor([ torch.norm(param, p=2)**2 for param in model.parameters()])
        return torch.mean(parameters)
    
    def _compute_average_param_grad_norm(self, model: torch.nn.Module) -> torch.Tensor:
        parameters_grad = torch.tensor([ torch.norm(param.grad, p=2)**2 for param in model.parameters()])
        return torch.mean(parameters_grad)

    def _compute_l2_reg(self, model: torch.nn.Module) -> torch.Tensor:
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)**2
        return l2_reg

class CompatibleActorCriticAgent(ActorCriticAgent):
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
            actor_critic = ActorCriticModule(LinearModule(), LinearModule())
        else:
            warnings.warn(
                "Using custom actor critic, please ensure compatible parameter space of actor and critic.",
                UserWarning,
                stacklevel=2
            )
        self.actor_critic = actor_critic

        self.avg_reward = 0

        # Load hyperparameters from config
        self.alpha = config['alpha']
        self.lr_critic = config['lr_critic']
        self.lr_actor = config['lr_actor']

        self.last_action = None
        self.debug = True
    
    def step(self, state: State, output_probs: bool = False) -> Union[Action, torch.Tensor]:
        # Make sure we're in eval mode for inference
        self.actor_critic.eval()
        
        with torch.no_grad():
            logits = self.actor_critic.actor(state)

            # print('logits', logits) if self.debug else None
            action_dist = Bernoulli(logits=logits)
            sampled_action = action_dist.sample()
            self.last_action = sampled_action
        
        # Switch back to train mode
        self.actor_critic.train()
        return sampled_action.item() if not output_probs else action_dist.probs

    def update(self, state: State, next_state: State, reward: float):        
        self.actor_critic.zero_grad()
        
        # Compute td
        with torch.no_grad():
            value_state = self.actor_critic.critic(state)
            value_next_state_detached = self.actor_critic.critic(next_state).detach()
            td = reward - self.avg_reward + value_next_state_detached - value_state
            # Update average reward
            self.avg_reward += self.alpha * td
        
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
                param_w += self.lr_critic * (td - dot_product) * phi_component
        
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
        
if __name__ == '__main__':
    # Set seed for reproducibility
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    env = Env(seed=seed_value)
    agent = ExhaustiveAgent(seed=seed_value)
    agent.offline_update(env)