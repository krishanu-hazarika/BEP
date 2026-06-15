from typing import List, Tuple
from game import (Card, create_deck, shuffle_deck, draw_cards, replace_cards, evaluate_hand, all_keep_actions,)
import time

def simulate_action(initial_hand: List[Card], keep_indices: List[int], num_simulations: int = 1000, scoring_system: str = "linear",) -> float:
    """
    Estimate expected reward of one keep action using Monte Carlo (MC) simulation.
    """
    total_score = 0.0

    for _ in range(num_simulations):
        deck = create_deck()

        for card in initial_hand:
            deck.remove(card)

        shuffle_deck(deck)

        final_hand = replace_cards(initial_hand, keep_indices, deck) # generating the final hand after replacing discarded cards
        score, _ = evaluate_hand(final_hand, scoring_system=scoring_system) # evaluating the resulting hand under the selected scoring system
        total_score += score # adding the obtained reward to the cumulative simulation score

    return total_score / num_simulations # returning the estimated expected reward across all simulations

def best_oracle_action(initial_hand: List[Card], num_simulations: int = 1000, scoring_system: str = "linear", return_timing: bool = False):
    """
    Evaluate all possible keep/discard actions using MC simulation and selecting the action with the highest estimated expected reward.
    It also returns timing details for action generation and simulation.
    """
    timing = {"oracle_action_generation_time": 0.0, "oracle_simulation_time": 0.0,}

    start = time.time()
    actions = all_keep_actions(len(initial_hand))
    timing["oracle_action_generation_time"] += time.time() - start

    best_action = None
    best_value = float("-inf")

    for action in actions:
        start = time.time()
        estimated_value = simulate_action(initial_hand, action, num_simulations, scoring_system,)
        timing["oracle_simulation_time"] += time.time() - start


        if estimated_value > best_value:
            best_value = estimated_value
            best_action = action
    
    if return_timing:
        return best_action, best_value, timing

    return best_action, best_value
    