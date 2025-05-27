import gymnasium as gym
import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from typing import List, Tuple, Dict, Any, Optional
from minimal_pg.a2c import ActorCriticAgent, ActorCriticConfig
from minimal_pg.nn_module import MLPModule
from minimal_pg.base import OnlineAgent
import time
import wandb

def run_episode(
    env: gym.Env,
    agent: OnlineAgent,
    max_steps: int = 1000,
    render: bool = False,
    seed: Optional[int] = None
) -> Tuple[float, int]:
    """
    Run a single episode of the environment.
    
    Args:
        env: Gymnasium environment
        agent: RL agent
        max_steps: Maximum number of steps per episode
        render: Whether to render the environment
        seed: Optional seed for environment reset
        
    Returns:
        Tuple of (total reward, number of steps)
    """
    state, info = env.reset(seed=seed)
    total_reward = 0
    done = False
    truncated = False
    step = 0
    
    while not (done or truncated) and step < max_steps:
        action = agent.step(state)
        next_state, reward, done, truncated, info = env.step(action)
        agent.update(state, next_state, reward, done)
        state = next_state
        total_reward += reward
        step += 1
        
        if render:
            env.render()
            time.sleep(0.01)  # Add small delay for visualization
            
    return total_reward, step

def benchmark_agent(
    env_name: str,
    agent: OnlineAgent,
    num_episodes: int = 100,
    max_steps: int = 1000,
    eval_interval: int = 10,
    render: bool = False,
    seed: Optional[int] = None,
    use_wandb: bool = False,
    project_name: str = "minimal-actor-critic",
    wandb_config: Optional[Dict[str, Any]] = None
) -> Tuple[List[float], List[float]]:
    """
    Benchmark an agent on a Gymnasium environment.
    
    Args:
        env_name: Name of the Gymnasium environment
        agent: RL agent to benchmark
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        eval_interval: How often to evaluate performance
        render: Whether to render the environment
        seed: Optional seed for reproducibility
        use_wandb: Whether to log metrics to Weights & Biases
        project_name: Name of the wandb project
        wandb_config: Additional configuration for wandb
        
    Returns:
        Tuple of (training rewards, evaluation rewards)
    """
    # Initialize wandb if requested
    if use_wandb:
        config = {
            "env_name": env_name,
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "eval_interval": eval_interval,
            "seed": seed
        }
        if wandb_config:
            config.update(wandb_config)
        wandb.init(project=project_name, config=config)
    
    env = gym.make(env_name)
    training_rewards = []
    eval_rewards = []
    
    # Track average rewards
    avg_training_reward = 0.0
    avg_eval_reward = 0.0
    window_size = 100  # Size of window for moving average
    
    for episode in range(num_episodes):
        # Training episode
        reward, steps = run_episode(env, agent, max_steps, render=False, seed=seed)
        training_rewards.append(reward)
        
        # Update average training reward
        if episode < window_size:
            avg_training_reward = sum(training_rewards) / (episode + 1)
        else:
            avg_training_reward = sum(training_rewards[-window_size:]) / window_size
        
        # Evaluation episode
        if episode % eval_interval == 0:
            eval_reward, _ = run_episode(env, agent, max_steps, render=render, seed=seed)
            eval_rewards.append(eval_reward)
            
            # Update average evaluation reward
            avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
            
            # Log episode metrics to wandb
            if use_wandb and isinstance(agent, ActorCriticAgent) and agent.use_wandb:
                wandb.log({
                    "benchmark/episode": episode,
                    "benchmark/training_reward": reward,
                    "benchmark/avg_training_reward": avg_training_reward,
                    "benchmark/eval_reward": eval_reward,
                    "benchmark/avg_eval_reward": avg_eval_reward,
                    "benchmark/steps": steps
                })
            
            print(f"Episode {episode}: Training Reward = {reward:.2f}, Avg Training Reward = {avg_training_reward:.2f}, "
                  f"Eval Reward = {eval_reward:.2f}, Avg Eval Reward = {avg_eval_reward:.2f}")
    
    env.close()
    
    if use_wandb:
        wandb.finish()
    
    return training_rewards, eval_rewards

def plot_results(
    training_rewards: List[float],
    eval_rewards: List[float],
    title: str = "Training and Evaluation Rewards",
    window_size: int = 10
) -> None:
    """
    Plot training and evaluation rewards.
    
    Args:
        training_rewards: List of training rewards
        eval_rewards: List of evaluation rewards
        title: Plot title
        window_size: Size of moving average window
    """
    plt.figure(figsize=(10, 6))
    
    # Plot raw training rewards
    plt.plot(training_rewards, alpha=0.3, label='Raw Training Rewards')
    
    # Plot moving average of training rewards
    if len(training_rewards) >= window_size:
        moving_avg = np.convolve(training_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(training_rewards)), moving_avg, label=f'Training Rewards (MA-{window_size})')
    
    # Plot evaluation rewards
    eval_x = np.arange(0, len(training_rewards), len(training_rewards)//len(eval_rewards))[:len(eval_rewards)]
    plt.plot(eval_x, eval_rewards, 'r-', label='Evaluation Rewards')
    
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environment
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    
    # Create agent with custom configuration
    config = ActorCriticConfig(
        hidden_dim=128,
        lr_actor=0.001,
        lr_critic=0.001,
        gamma=0.99,
        lambda_v=0.01,
        lambda_entropy=0.01,
        grad_clip_norm=1.0
    )
    
    # Create wandb config
    wandb_config = {
        "hidden_dim": config.hidden_dim,
        "lr_actor": config.lr_actor,
        "lr_critic": config.lr_critic,
        "gamma": config.gamma,
        "lambda_v": config.lambda_v,
        "lambda_entropy": config.lambda_entropy,
        "grad_clip_norm": config.grad_clip_norm,
        "seed": seed
    }
    
    agent = ActorCriticAgent(
        env, 
        config=config, 
        seed=seed, 
        use_wandb=True
    )
    
    # Run benchmark
    training_rewards, eval_rewards = benchmark_agent(
        env_name=env_name,
        agent=agent,
        num_episodes=10_000,
        max_steps=1_000,
        eval_interval=100,
        render=True,
        seed=seed,
        use_wandb=True,
        project_name="minimal-actor-critic",
        wandb_config=wandb_config
    )
    
    # Plot results
    plot_results(training_rewards, eval_rewards, title=f"A2C Performance on {env_name}")

if __name__ == "__main__":
    main()