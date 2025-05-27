import matplotlib.pyplot as plt
import numpy as np
from agent import ActorCriticAgent
from env import Env
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

def visualize_policy_heatmap(agent: ActorCriticAgent, env: Env):
    states = env.state_set
    
    actions = np.array([round(agent.step(state, output_probs=True).item(), 2) for state in states])
    
    # Reshape actions to a 2D grid for heatmap
    action_grid = actions.reshape((env.total_servers+1, len(env.priorities)))
    
    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(action_grid, cmap='coolwarm', aspect='auto')
    plt.colorbar(im, label='Action')
    
    # Set ticks at the center of each cell
    y_ticks = np.arange(env.total_servers + 1)
    x_ticks = np.arange(len(env.priorities))
    plt.yticks(y_ticks, y_ticks)
    plt.xticks(x_ticks, env.priorities)  # Use actual priority values as x-tick labels
    
    plt.xlabel('Priority')
    plt.ylabel('Free Servers')
    plt.title('Policy Heatmap')
    
    # Move ticks to the center of each cell
    ax = plt.gca()
    ax.set_xticks(x_ticks - 0.5, minor=True)
    ax.set_yticks(y_ticks - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    plt.close(fig)
    return fig

def print_policy(agent: ActorCriticAgent, env: Env):
    states = env.state_set
    actions = np.array([round(agent.step(state, output_probs=True).item(), 2) for state in states])
    return actions

def plot_actor_critic_vs_exact_policy_gradient(
    results_path: Path,
    optimal_baseline: float = 2.680,
    figsize: Tuple[int, int] = (10, 6),
    confidence_level: float = 0.95,
    num_bootstrap_samples: int = 10000
) -> None:
    results = pd.read_csv(results_path)
    steps = results['step'].values
    
    # Get columns for each method
    ac_columns = [col for col in results.columns if col.startswith('actor_critic_e')]
    
    # Calculate means and CIs for actor-critic
    ac_means, ac_lower_cis, ac_upper_cis = [], [], []
    for step in steps:
        step_values = results.loc[results['step'] == step, ac_columns].values.flatten()
        ci = bs.bootstrap(
            step_values,
            stat_func=bs_stats.mean,
            num_iterations=num_bootstrap_samples,
            alpha=1-confidence_level
        )
        ac_means.append(ci.value)
        ac_lower_cis.append(ci.lower_bound)
        ac_upper_cis.append(ci.upper_bound)
    
    # Get exact policy gradient values
    epg_values = results['exact_policy_gradient'].values
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot optimal baseline
    plt.axhline(y=optimal_baseline, color='blue', linestyle='-', label='Optimal Baseline')
    
    # Plot actor-critic with CI
    plt.plot(steps, ac_means, color='red', linestyle='--', label='Actor-Critic')
    plt.fill_between(steps, ac_lower_cis, ac_upper_cis, color='red', alpha=0.2)
    
    # Plot exact policy gradient
    plt.plot(steps, epg_values, color='red', linestyle='-', label='Exact Policy Gradient')
    
    plt.legend(loc='lower right')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Algorithm Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "task3.png", bbox_inches='tight')
    plt.savefig(output_dir / "task3.pdf", bbox_inches='tight')
    plt.close()

def plot_all_results(
    results_path: Path,
    optimal_baseline: float = 2.680,
    figsize: Tuple[int, int] = (10, 6),
    confidence_level: float = 0.95,
    num_bootstrap_samples: int = 10000
) -> None:
    """
    Plot all results including actor-critic, compatible actor-critic, exact policy gradient,
    and compatible exact policy gradient with bootstrapped confidence intervals.
    
    Args:
        results_path: Path to the results CSV file
        optimal_baseline: The optimal baseline value to plot
        figsize: Figure size
        confidence_level: Confidence level for the bootstrap intervals
        num_bootstrap_samples: Number of bootstrap samples to generate
    """
    results = pd.read_csv(results_path)
    steps = results['step'].values
    
    # Get columns for each method
    ac_columns = [col for col in results.columns if col.startswith('actor_critic_e')]
    cac_columns = [col for col in results.columns if col.startswith('compatible_actor_critic_e')]
    
    # Calculate means and CIs for actor-critic
    ac_means, ac_lower_cis, ac_upper_cis = [], [], []
    for step in steps:
        step_values = results.loc[results['step'] == step, ac_columns].values.flatten()
        ci = bs.bootstrap(
            step_values,
            stat_func=bs_stats.mean,
            num_iterations=num_bootstrap_samples,
            alpha=1-confidence_level
        )
        ac_means.append(ci.value)
        ac_lower_cis.append(ci.lower_bound)
        ac_upper_cis.append(ci.upper_bound)
    
    # Calculate means and CIs for compatible actor-critic
    cac_means, cac_lower_cis, cac_upper_cis = [], [], []
    for step in steps:
        step_values = results.loc[results['step'] == step, cac_columns].values.flatten()
        ci = bs.bootstrap(
            step_values,
            stat_func=bs_stats.mean,
            num_iterations=num_bootstrap_samples,
            alpha=1-confidence_level
        )
        cac_means.append(ci.value)
        cac_lower_cis.append(ci.lower_bound)
        cac_upper_cis.append(ci.upper_bound)
    
    # Get exact policy gradient values
    epg_values = results['exact_policy_gradient'].values
    cepg_values = results['compatible_exact_policy_gradient'].values
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot optimal baseline
    plt.axhline(y=optimal_baseline, color='blue', linestyle='-', label='Optimal Baseline')
    
    # Plot actor-critic with CI
    plt.plot(steps, ac_means, color='red', linestyle='--', label='Actor-Critic')
    plt.fill_between(steps, ac_lower_cis, ac_upper_cis, color='red', alpha=0.2)
    
    # Plot compatible actor-critic with CI
    plt.plot(steps, cac_means, color='green', linestyle='--', label='Compatible Actor-Critic')
    plt.fill_between(steps, cac_lower_cis, cac_upper_cis, color='green', alpha=0.1)
    
    # Plot exact policy gradient
    plt.plot(steps, epg_values, color='red', linestyle='-', label='Exact Policy Gradient')
    
    # Plot compatible exact policy gradient
    plt.plot(steps, cepg_values, color='green', linestyle='-', label='Compatible Exact Policy Gradient')
    
    plt.legend(loc='lower right')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Algorithm Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "task4.png", bbox_inches='tight')
    plt.savefig(output_dir / "task4.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_actor_critic_vs_exact_policy_gradient(
        results_path=Path("results/experiment_results.csv"),
        optimal_baseline=2.680
    )

    plot_all_results(
        results_path=Path("results/experiment_results.csv"),
        optimal_baseline=2.680
    )