import math
import time
import numpy as np
from typing import List
from game import Card, all_keep_actions
from oracle import simulate_action

def ucb_select_action(action_counts: np.ndarray, action_values: np.ndarray, total_pulls: int, exploration_constant: float = 2.0) -> int:
    """
    Selects an action index using the upper confidence bound (UCB) rule.
    """
    zero_mask = action_counts == 0 

    if np.any(zero_mask):
        return int(np.flatnonzero(zero_mask)[0])

    bonus = exploration_constant * np.sqrt(np.log(total_pulls) / action_counts)
    ucb_values = action_values + bonus
    return int(np.argmax(ucb_values))

def best_bandit_action(initial_hand: List[Card], num_rounds: int = 100, simulation_budget_per_pull: int = 1, exploration_constant: float = 2.0, scoring_system: str = "linear", return_timing: bool = False):
    """
    Uses a UCB-style bandit search to identify a strong discard action for one hand.
    It also returns timing details for UCB selection, simulation, and value updates.
    """

    timing = {"bandit_action_generation_time": 0.0, "bandit_ucb_selection_time": 0.0, "bandit_simulation_time": 0.0, "bandit_update_time": 0.0,}
    
    start = time.time()
    actions = all_keep_actions(len(initial_hand)) # generating all possible keep/discard actions for the current hand size
    timing["bandit_action_generation_time"] += time.time() - start
    num_actions = len(actions) # storing the total number of possible actions

    action_counts = np.zeros(num_actions, dtype=np.int32) # tracking the number of times each action has been selected during bandit search
    action_values = np.zeros(num_actions, dtype=np.float32)  # storing the running average reward estimate for each action

    total_pulls = 0 # counting the total number of action selections performed during the search

    for _ in range(num_rounds):
        start = time.time()
        action_idx = ucb_select_action(action_counts, action_values, max(1, total_pulls), exploration_constant) # selecting the next action to explore using the UCB bandit strategy
        timing["bandit_ucb_selection_time"] += time.time() - start

        start = time.time()
        action = actions[action_idx] # retrieving the discard action corresponding to the selected action index
        reward = simulate_action(initial_hand, action, num_simulations=simulation_budget_per_pull, scoring_system=scoring_system,) # estimating the expected reward of the selected action through MC simulation
        timing["bandit_simulation_time"] += time.time() - start

        start = time.time()
        total_pulls += 1
        action_counts[action_idx] += 1
        n = action_counts[action_idx] # updating the estimated average reward of the selected action incrementally without storing all previous simulation rewards in memory
        old_mean = action_values[action_idx] # retrieving the previous estimated mean reward for this action
        new_mean = old_mean + (reward - old_mean) / n # computing the updated running average after observing the new reward
        action_values[action_idx] = new_mean # storing the updated estimated value for the action
        timing["bandit_update_time"] += time.time() - start

    best_idx = int(np.argmax(action_values)) # selecting the action with the highest estimated average reward after all bandit search rounds are completed
    
    if return_timing:
        return actions[best_idx], action_values[best_idx], timing
    
    return actions[best_idx], action_values[best_idx]