# TODO: Clean and Integrate with benchmark.py

from minimal_pg.base import BaseAgent
from minimal_pg.__agent import ActorCriticModule
from minimal_pg.env import Action, State
from minimal_pg.nn_module import MLPModule, LinearModule
import torch
import random
import numpy as np
import yaml
import warnings

class CompatibleActorCriticAgent(BaseAgent):
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