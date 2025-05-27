from minimal_pg.nn_module import MLPModule, LinearModule
from minimal_pg.base import OnlineAgent
import torch
import random
import numpy as np
import warnings
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from torch.distributions import Categorical, Bernoulli
from typing import Union, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import wandb

from minimal_pg.base import State, Action, Reward

@dataclass
class ActorCriticConfig:
    """Configuration for Actor-Critic agent."""
    hidden_dim: int = 64
    lr_actor: float = 0.0001
    lr_critic: float = 0.0001
    gamma: float = 0.99  # Discount factor for future rewards
    lambda_v: float = 1  # L2 regularization coefficient for critic
    lambda_entropy: float = 0.01  # Entropy regularization coefficient
    grad_clip_norm: float = 1.0  # Gradient clipping norm


# TODO: Remove this class, increasing inheritance complexity.
# 1. Directly using nn_module.MLPModule or nn_module.LinearModule
# 2. Remove ActorCriticConfig.hidden_dim that assuming using nn_module.MLPModule
class ActorCriticModule(torch.nn.Module): 
    """
    Neural network module that combines actor and critic networks.
    Actor outputs action logits, critic outputs state values.
    """
    def __init__(self, actor: torch.nn.Module, critic: torch.nn.Module):
        super().__init__()
        self.actor_model = actor
        self.critic_model = critic
    
    def actor(self, state: State) -> torch.Tensor:
        """
        Forward pass through actor network.
        
        Args:
            state: Environment state, can be array or scalar
            
        Returns:
            Action logits tensor
        """
        return self.actor_model(state)

    def critic(self, state: State) -> torch.Tensor:
        """
        Forward pass through critic network.
        
        Args:
            state: Environment state, can be array or scalar
            
        Returns:
            State value tensor
        """
        return self.critic_model(state)

class ActorCriticAgent(OnlineAgent):
    """
    Advantage Actor-Critic (A2C) agent implementation.
    Supports discrete action spaces and both discrete/continuous observation spaces.
    """
    def __init__(
        self, 
        env: gym.Env,
        config: Optional[ActorCriticConfig] = None,
        actor_critic: Optional[ActorCriticModule] = None, 
        seed: int = 42,
        use_wandb: bool = False
    ):
        """
        Initialize A2C agent.
        
        Args:
            env: Gymnasium environment
            config: Optional configuration for the agent. If None, uses default values.
            actor_critic: Optional custom actor-critic module
            seed: Random seed for reproducibility
            use_wandb: Whether to log metrics to Weights & Biases
        """
        # Set the random seed
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Use default config if none provided
        self.config = config or ActorCriticConfig()
        
        # Determine observation and action dimensions from the environment
        if isinstance(env.observation_space, Box):
            # Assuming a 1D Box, or that the observation will be flattened before passing to the agent
            obs_dim = int(np.prod(env.observation_space.shape))
        elif isinstance(env.observation_space, Discrete):
            # If observation is a single integer, obs_dim is 1.
            # If one-hot encoding is used externally, obs_dim would be env.observation_space.n.
            # The nn_module's forward method now handles scalar inputs by unsqueezing.
            obs_dim = 1 # Assuming raw discrete observation is passed as a scalar
        else:
            raise NotImplementedError(f"Unsupported observation space: {env.observation_space}")

        if isinstance(env.action_space, Discrete):
            action_dim = env.action_space.n
            self.action_type = 'discrete'
        # elif isinstance(env.action_space, Box): # Continuous actions
        #     action_dim = env.action_space.shape[0]
        #     self.action_type = 'continuous'
        #     raise NotImplementedError("Continuous A2C not fully implemented here (requires Gaussian policy).")
        else:
            raise NotImplementedError(f"Unsupported action space: {env.action_space}. Only Discrete is supported.")

        if actor_critic is None:
            actor_net = LinearModule(obs_dim, action_dim)
            critic_net = LinearModule(obs_dim, 1)
            self.actor_critic = ActorCriticModule(actor_net, critic_net)
        else:
            self.actor_critic = actor_critic

        self.last_action: Optional[torch.Tensor] = None
        self.debug = True
        
        # Wandb logging setup
        self.use_wandb = use_wandb
        self.step_count = 0
    
    def step(self, state: State, output_probs: bool = False) -> Union[Action, torch.Tensor]:
        """
        Select action for given state.
        
        Args:
            state: Current environment state
            output_probs: If True, return action probabilities instead of action
            
        Returns:
            Selected action or action probabilities
        """
        self.actor_critic.eval()
        
        with torch.no_grad():
            logits = self.actor_critic.actor(state)

            if self.action_type == 'discrete':
                if logits.isnan().any():
                    print(f"state: {state}")
                    param_weights = [ param.detach().cpu().numpy() for param in self.actor_critic.actor_model.parameters()]
                    print(f"weights: {param_weights}")
                    print(f"logits: {logits}")
                    print(f"Categorical: {Categorical(logits=logits)}")
                    exit()
                action_dist = Categorical(logits=logits)
            else: # Should not happen due to __init__ check
                raise NotImplementedError("Only discrete actions are supported.")
            
            sampled_action_tensor = action_dist.sample()
            self.last_action = sampled_action_tensor # Store tensor for log_prob
            
            if self.use_wandb and wandb.run is not None:
                    # Log action distribution metrics
                wandb.log({
                    # "a2c/action_entropy": action_dist.entropy().item(),
                    "a2c/action_probs": wandb.Histogram(action_dist.probs.detach().cpu().numpy()),
                    "a2c/selected_action": sampled_action_tensor.item()
                }, step=self.step_count)
        
        self.actor_critic.train()
        # Return Python number if env expects it, else could return tensor if using batched envs later
        return sampled_action_tensor.item() if not output_probs else action_dist.probs

    def update(self, state: State, next_state: State, reward: Reward, done: bool):        
        """
        Update actor and critic networks using A2C algorithm.
        
        Args:
            state: Current state
            next_state: Next state
            reward: Reward received
        """
        self.actor_critic.zero_grad()
        
        # Compute td
        with torch.no_grad():
            value_state = self.actor_critic.critic(state)
            value_next_state_detached = self.actor_critic.critic(next_state).detach()
            # Calculate discounted TD error
            # Assuming reward is a scalar or a tensor that can be broadcasted.
            td_error = reward + self.config.gamma * (1 - done) * value_next_state_detached - value_state

        # Update critic
        # Recompute value_state for gradient tracking
        value_state_for_grad = self.actor_critic.critic(state)
        critic_reg = self.config.lambda_v * self._compute_l2_reg(self.actor_critic.critic_model)
        
        # Use TD squared error minimization for critic
        critic_target = reward + self.config.gamma * (1 - done) * value_next_state_detached
        critic_loss = torch.nn.functional.mse_loss(value_state_for_grad, critic_target.detach()) + critic_reg
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.critic_model.parameters(), 
            max_norm=self.config.grad_clip_norm
        )

        critic_grad_norm = self._compute_average_param_grad_norm(self.actor_critic.critic_model).detach()
        
        with torch.no_grad(): # Optimizer step
            for param in self.actor_critic.critic_model.parameters():
                if param.grad is not None:
                    param -= self.config.lr_critic * param.grad  # Note the negative sign for gradient descent
        
        self.actor_critic.zero_grad() # Zero gradients before actor update
        
        action_logits = self.actor_critic.actor(state)
        if self.action_type == 'discrete':
            dist = Categorical(logits=action_logits)
        else: 
            raise NotImplementedError("Only discrete actions are supported.")

        log_prob = dist.log_prob(self.last_action)
        obj_actor = log_prob + self.config.lambda_entropy * dist.entropy()
        obj_actor.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.actor_model.parameters(), 
            max_norm=self.config.grad_clip_norm
        )

        actor_grad_norm = self._compute_average_param_grad_norm(self.actor_critic.actor_model).detach()
        
        with torch.no_grad(): # Optimizer step
            for param in self.actor_critic.actor_model.parameters():
                if param.grad is not None:
                    param += self.config.lr_actor * td_error.detach() * param.grad
        
        if self.use_wandb and wandb.run is not None:
            # Log metrics
            metrics = {
                "a2c/reward": reward, # This is the current step reward
                "a2c/td_error": td_error.item(),
                "a2c/value_state": value_state.item(),
                "a2c/value_next_state": value_next_state_detached.item(),
                "a2c/critic_reg": critic_reg.item(),
                "a2c/critic_loss": critic_loss.item(),
                "a2c/obj_actor": obj_actor.item(),
                "a2c/log_prob": log_prob.item(),
                "a2c/entropy": dist.entropy().item(),
                "a2c/actor_grad_norm": actor_grad_norm,
                "a2c/critic_grad_norm": critic_grad_norm,
                "a2c/actor_param_norm": self._compute_average_param_norm(self.actor_critic.actor_model).item(),
                "a2c/critic_param_norm": self._compute_average_param_norm(self.actor_critic.critic_model).item()
            }
            wandb.log(metrics, step=self.step_count)
            self.step_count += 1
                    
    def _compute_l2_reg(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Compute L2 regularization term for model parameters.
        
        Args:
            model: Neural network model
            
        Returns:
            L2 regularization term
        """
        l2_reg = torch.tensor(0., dtype=torch.float32)
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)**2
        return l2_reg

    def _compute_average_param_norm(self, model: torch.nn.Module) -> torch.Tensor:
        parameters = torch.tensor([ torch.norm(param, p=2)**2 for param in model.parameters()])
        return torch.mean(parameters)
    
    def _compute_average_param_grad_norm(self, model: torch.nn.Module) -> torch.Tensor:
        parameters_grad = torch.tensor([ torch.norm(param.grad, p=2)**2 for param in model.parameters()])
        return torch.mean(parameters_grad)