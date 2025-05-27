import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from minimal_pg.base import OnlineAgent, State, Action

def visualize_policy(
    agent: OnlineAgent,
    env: gym.Env,
    num_states: int = 100,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Policy Visualization",
    cmap: str = "viridis"
) -> plt.Figure:
    """
    Visualize the policy's action probabilities across different states.
    Creates a heatmap where each cell (i,j) represents the probability of taking action j in state i.
    
    Args:
        agent: The policy to visualize
        env: The environment to sample states from
        num_states: Number of states to sample for visualization
        figsize: Figure size (width, height)
        title: Plot title
        cmap: Colormap for the heatmap
        
    Returns:
        matplotlib Figure object that can be logged to wandb
        
    Note:
        - For discrete observation spaces, samples states uniformly
        - For continuous observation spaces, samples states from a grid
        - Only works with discrete action spaces
    """
    if not isinstance(env.action_space, Discrete):
        raise NotImplementedError("Policy visualization only supports discrete action spaces")
    
    # Sample states
    if isinstance(env.observation_space, Discrete):
        # For discrete states, sample uniformly
        states = np.random.randint(0, env.observation_space.n, size=num_states)
    elif isinstance(env.observation_space, Box):
        # For continuous states, create a grid
        if len(env.observation_space.shape) != 1:
            raise NotImplementedError("Only 1D continuous observation spaces are supported")
        
        # Create a grid of states
        low = env.observation_space.low
        high = env.observation_space.high
        states = np.linspace(low, high, num_states)
    else:
        raise NotImplementedError(f"Unsupported observation space: {env.observation_space}")
    
    # Get action probabilities for each state
    action_probs = []
    for state in states:
        probs = agent.step(state, output_probs=True)
        if isinstance(probs, np.ndarray):
            action_probs.append(probs)
        else:
            # Convert tensor to numpy if needed
            action_probs.append(probs.detach().cpu().numpy())
    
    action_probs = np.array(action_probs)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(action_probs, aspect='auto', cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Action Probability", rotation=-90, va="bottom")
    
    # Set labels and title
    ax.set_xlabel("Action")
    ax.set_ylabel("State")
    ax.set_title(title)
    
    # Set ticks
    ax.set_xticks(np.arange(env.action_space.n))
    ax.set_yticks(np.arange(num_states))
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def log_policy_to_wandb(
    agent: OnlineAgent,
    env: gym.Env,
    wandb,
    step: int,
    num_states: int = 100,
    title: str = "Policy Visualization"
) -> None:
    """
    Create and log a policy visualization to Weights & Biases.
    
    Args:
        agent: The policy to visualize
        env: The environment to sample states from
        wandb: Weights & Biases instance
        step: Current training step
        num_states: Number of states to sample for visualization
        title: Plot title
    """
    fig = visualize_policy(agent, env, num_states=num_states, title=title)
    wandb.log({title: wandb.Image(fig)}, step=step)
    plt.close(fig)
