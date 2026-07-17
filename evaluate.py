import time
import random
import csv
import numpy as np
from pathlib import Path
from game import create_deck, shuffle_deck, draw_cards
from oracle import best_oracle_action, simulate_action
from greedy import greedy_keep_indices
from bandit import best_bandit_action
from qlearning import QLearningAgent
from scipy.stats import ttest_rel

random.seed(42) # setting a fixed random seed to ensure reproducible and consistent experimental results across multiple runs
np.random.seed(42)

def evaluate_strategies(num_hands: int = 50, hand_size: int = 5, scoring_system: str = "video_poker", oracle_simulations: int = 1000, bandit_rounds: int = 300, bandit_budget_per_pull: int = 3, bandit_exploration_constant: float = 2.0, q_episodes: int = 20000,):
    """
    Evaluates oracle search, greedy heuristics, bandit exploration, and Q-learning across multiple randomly generated draw poker hands under a selected scoring system.
    
    The function first trains a Q-learning agent for the specified hand size and reward structure. It then repeatedly generates random initial hands and applies
    all four strategies to the same hand.
    For each hand:
    - Greedy search selects a heuristic keep/discard action.
    - Bandit search selects an action using UCB-guided adaptive sampling.
    - Q-learning selects the learned best action from the trained Q-table.
    - Oracle search exhaustively evaluates all actions using Monte Carlo simulation.
    
    Each selected action is evaluated using MC simulation to estimate its expected reward under the chosen scoring system.
    
    The function records:
    - average expected reward for each strategy,
    - reward standard deviation,
    - regret relative to oracle performance,
    - agreement rate with oracle discard decisions,
    - total experimental runtime.
    
    Returns a dictionary containing all aggregated evaluation metrics and
    experimental configuration parameters.
    """
    oracle_values = []
    greedy_values = []
    bandit_values = []
    q_values = []

    greedy_same_as_oracle = 0
    bandit_same_as_oracle = 0
    q_same_as_oracle = 0

    greedy_runtime = 0.0
    bandit_runtime = 0.0
    qlearning_runtime = 0.0
    oracle_runtime = 0.0

    bandit_action_generation_time = 0.0
    bandit_ucb_selection_time = 0.0
    bandit_simulation_time = 0.0
    bandit_update_time = 0.0

    q_inference_state_time = 0.0
    q_inference_lookup_time = 0.0

    oracle_action_generation_time = 0.0
    oracle_simulation_time = 0.0

    print(f"\nRunning: n={hand_size}, scoring={scoring_system}")
    print("Training Q-learning agent...")

    q_training_start = time.time()
    q_agent = QLearningAgent(hand_size=hand_size, scoring_system=scoring_system, gamma=0.0, epsilon=0.1,) # initializing the Q-learning agent with the selected game configuration and learning parameters
    q_training_details = q_agent.train(num_episodes=q_episodes, return_timing=True,) # training the Q-learning agent over many simulated episodes to learn action values
    q_training_runtime = time.time() - q_training_start

    start_time = time.time() # recording the starting time of the experiment to measure total runtime

    for _ in range(num_hands):
        deck = create_deck()
        shuffle_deck(deck)
        initial_hand = draw_cards(deck, hand_size)

        strategy_start = time.time()
        greedy_action = tuple(sorted(greedy_keep_indices(initial_hand))) # normalized to a sorted tuple so it is directly comparable with the oracle's action
        greedy_value = simulate_action(initial_hand, greedy_action, num_simulations=oracle_simulations, scoring_system=scoring_system,)
        greedy_runtime += time.time() - strategy_start

        strategy_start = time.time()
        bandit_action, _, bandit_timing = best_bandit_action(initial_hand, num_rounds=bandit_rounds, simulation_budget_per_pull=bandit_budget_per_pull, exploration_constant=bandit_exploration_constant, scoring_system=scoring_system, return_timing=True,)
        bandit_action = tuple(sorted(bandit_action))
        bandit_value = simulate_action(initial_hand, bandit_action, num_simulations=oracle_simulations, scoring_system=scoring_system,)
        bandit_runtime += time.time() - strategy_start
        bandit_action_generation_time += bandit_timing["bandit_action_generation_time"]
        bandit_ucb_selection_time += bandit_timing["bandit_ucb_selection_time"]
        bandit_simulation_time += bandit_timing["bandit_simulation_time"]
        bandit_update_time += bandit_timing["bandit_update_time"]
        
        strategy_start = time.time()
        q_action, _, q_inference_timing = q_agent.best_action(initial_hand, return_timing=True,)
        q_action = tuple(sorted(q_action))
        q_value = simulate_action(initial_hand, q_action, num_simulations=oracle_simulations, scoring_system=scoring_system,)
        qlearning_runtime += time.time() - strategy_start
        q_inference_state_time += q_inference_timing["q_inference_state_time"]
        q_inference_lookup_time += q_inference_timing["q_inference_lookup_time"]

        strategy_start = time.time()
        oracle_action, oracle_value, oracle_timing = best_oracle_action(initial_hand, num_simulations=oracle_simulations, scoring_system=scoring_system, return_timing=True,)
        oracle_action = tuple(sorted(oracle_action))
        oracle_runtime += time.time() - strategy_start
        oracle_action_generation_time += oracle_timing["oracle_action_generation_time"]
        oracle_simulation_time += oracle_timing["oracle_simulation_time"]

        greedy_values.append(greedy_value) # storing the greedy expected reward 
        bandit_values.append(bandit_value) # storing the bandit expected reward 
        q_values.append(q_value) # storing the Q-learning expected reward 
        oracle_values.append(oracle_value) # storing the oracle expected reward 

        if greedy_action == oracle_action:
            greedy_same_as_oracle += 1 # counting cases where greedy selected the same action as the oracle

        if bandit_action == oracle_action:
            bandit_same_as_oracle += 1 # counting cases where bandit search matched the oracle decision

        if q_action == oracle_action:
            q_same_as_oracle += 1 # counting cases where Q-learning matched the oracle decision

    runtime = time.time() - start_time # computing the total runtime of the experimental evaluation

    # statistical significance testing
    oracle_vs_greedy_p = float(ttest_rel(oracle_values, greedy_values).pvalue)
    oracle_vs_bandit_p = float(ttest_rel(oracle_values, bandit_values).pvalue)
    oracle_vs_qlearning_p = float(ttest_rel(oracle_values, q_values).pvalue)
    bandit_vs_greedy_p = float(ttest_rel(bandit_values, greedy_values).pvalue)
    bandit_vs_qlearning_p = float(ttest_rel(bandit_values, q_values).pvalue)
    greedy_vs_qlearning_p = float(ttest_rel(greedy_values, q_values).pvalue)

    results = {
        "hand_size": hand_size,
        "scoring_system": scoring_system,
        "num_hands": num_hands,
        "oracle_simulations": oracle_simulations,
        "bandit_rounds": bandit_rounds,
        "bandit_budget_per_pull": bandit_budget_per_pull,
        "bandit_exploration_constant": bandit_exploration_constant,
        "q_episodes": q_episodes,
        "avg_greedy": float(np.mean(greedy_values)),
        "avg_bandit": float(np.mean(bandit_values)),
        "avg_qlearning": float(np.mean(q_values)),
        "avg_oracle": float(np.mean(oracle_values)),
        "std_greedy": float(np.std(greedy_values, ddof=1)),
        "std_bandit": float(np.std(bandit_values, ddof=1)),
        "std_qlearning": float(np.std(q_values, ddof=1)),
        "std_oracle": float(np.std(oracle_values, ddof=1)),
        "regret_greedy": float(np.mean(oracle_values) - np.mean(greedy_values)),
        "regret_bandit": float(np.mean(oracle_values) - np.mean(bandit_values)),
        "regret_qlearning": float(np.mean(oracle_values) - np.mean(q_values)),
        "greedy_match_rate": greedy_same_as_oracle / num_hands,
        "bandit_match_rate": bandit_same_as_oracle / num_hands,
        "qlearning_match_rate": q_same_as_oracle / num_hands,
        
        "p_oracle_vs_greedy": oracle_vs_greedy_p,
        "p_oracle_vs_bandit": oracle_vs_bandit_p,
        "p_oracle_vs_qlearning": oracle_vs_qlearning_p,
        "p_bandit_vs_greedy": bandit_vs_greedy_p,
        "p_bandit_vs_qlearning": bandit_vs_qlearning_p,
        "p_greedy_vs_qlearning": greedy_vs_qlearning_p,
        
        "runtime_seconds": runtime,
        "greedy_runtime_seconds": greedy_runtime,
        "bandit_runtime_seconds": bandit_runtime,
        "qlearning_runtime_seconds": qlearning_runtime,
        "oracle_runtime_seconds": oracle_runtime,
        "q_training_runtime_seconds": q_training_runtime,
        "q_training_state_time": q_training_details["q_training_state_time"],
        "q_training_action_selection_time": q_training_details["q_training_action_selection_time"],
        "q_training_environment_time": q_training_details["q_training_environment_time"],
        "q_training_reward_time": q_training_details["q_training_reward_time"],
        "q_training_update_time": q_training_details["q_training_update_time"],
        "bandit_action_generation_time": bandit_action_generation_time,
        "bandit_ucb_selection_time": bandit_ucb_selection_time,
        "bandit_simulation_time": bandit_simulation_time,
        "bandit_update_time": bandit_update_time,
        "q_inference_state_time": q_inference_state_time,
        "q_inference_lookup_time": q_inference_lookup_time,
        "oracle_action_generation_time": oracle_action_generation_time,
        "oracle_simulation_time": oracle_simulation_time,
        "total_runtime_including_training_seconds": runtime + q_training_runtime,
    }

    return results

def save_results_to_csv(results: dict, output_folder: Path) -> None:
    """
    Saves aggregated experimental evaluation results to a CSV file.

    The filename is automatically generated using the hand size and
    scoring system configuration. The results dictionary is written
    as a single-row CSV table containing all recorded evaluation metrics
    and experimental parameters.
    """
    filename = f"evaluation_n{results['hand_size']}_{results['scoring_system']}.csv"
    filepath = output_folder / filename

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print(f"Saved: {filepath}")

if __name__ == "__main__":
    output_folder = Path("evaluation")
    output_folder.mkdir(exist_ok=True) # creating a folder called "evaluation" to store all results

    hand_sizes = [5, 6, 7] # defining the hand sizes evaluated in the experiments
    scoring_systems = ["video_poker"] 

    for scoring_system in scoring_systems:
        for hand_size in hand_sizes:
            filename = f"evaluation_n{hand_size}_{scoring_system}.csv"
            filepath = output_folder / filename
            
            if filepath.exists():
                print(f"Skipping existing file: {filename}")
                continue

            results = evaluate_strategies(num_hands=500, hand_size=hand_size, scoring_system=scoring_system, oracle_simulations=2000, bandit_rounds=500, bandit_budget_per_pull=5, bandit_exploration_constant=2.0, q_episodes=1000000,)

            save_results_to_csv(results, output_folder) # saving the aggregated evaluation metrics to a CSV file

    print("\nAll evaluations completed.")