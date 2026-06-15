import math
from typing import List, Tuple
from game import Card, all_keep_actions
from oracle import simulate_action
import time


def ucb_select_action(action_counts: List[int], action_values: List[float], total_pulls: int, exploration_constant: float = 2.0) -> int:
    """
    Selects an action index using the upper confidence bound (UCB) rule.
    """
    for i, count in enumerate(action_counts): # using a "for" loop to make sure every action is tried at least once
        if count == 0:
            return i

    best_index = 0
    best_ucb = float("-inf")

    for i in range(len(action_counts)): # iterating through all actions to compute their UCB scores and selecting the action with the highest exploration-exploitation value
        bonus = exploration_constant * math.sqrt(math.log(total_pulls) / action_counts[i])
        ucb_value = action_values[i] + bonus

        if ucb_value > best_ucb:
            best_ucb = ucb_value
            best_index = i

    return best_index

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

    action_counts = [0] * num_actions # tracking the number of times each action has been selected during bandit search
    action_values = [0.0] * num_actions  # storing the running average reward estimate for each action

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

    best_idx = max(range(num_actions), key=lambda i: action_values[i]) # selecting the action with the highest estimated average reward after all bandit search rounds are completed
    
    if return_timing:
        return actions[best_idx], action_values[best_idx], timing
    
    return actions[best_idx], action_values[best_idx]