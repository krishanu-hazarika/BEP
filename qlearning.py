import random
from collections import defaultdict
from typing import Dict, List, Tuple
from game import (Card, create_deck, shuffle_deck, draw_cards, replace_cards, hand_category_score, state_features, all_keep_actions,)
import time

class QLearningAgent:
    def __init__(self, hand_size: int = 5, scoring_system: str = "linear", alpha: float = 0.1, gamma: float = 0.0, epsilon: float = 0.1,):
        """
        alpha = Learning rate controlling how strongly new rewards update old Q-values.
        gamma = 0 because each episode is a one-step decision: initial hand -> choose to keep/discard -> final reward.
        epsilon = Exploration probability controlling how often random actions are selected.
        """
        self.hand_size = hand_size
        self.scoring_system = scoring_system
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions: List[List[int]] = all_keep_actions(hand_size) # generating all possible keep/discard actions for the selected hand size
        self.q_table: Dict[tuple, List[float]] = defaultdict(lambda: [0.0 for _ in range(len(self.actions))]) # creating a Q-table that maps each state to a list of estimated action values

    def choose_action(self, state: tuple) -> int:
        """
        Epsilon-greedy action selection.
        Returns the action index.
        """
        # exploring by selecting a random action with probability epsilon
        if random.random() < self.epsilon:
            return random.randrange(len(self.actions))

        q_values = self.q_table[state] # retrieving the estimated Q-values for the current state
        max_q = max(q_values)
        best_indices = [i for i, q in enumerate(q_values) if q == max_q] # finding all action indices that share the highest Q-value

        return random.choice(best_indices) # randomly selecting one of the best actions to break ties fairly

    def update(self, state: tuple, action_idx: int, reward: float, next_state: tuple = None,) -> None:
        """
        Q-learning update.
        Since this is a one-step episode, next state's future value is usually zero.
        """
        current_q = self.q_table[state][action_idx] # retrieving the current Q-value estimate for the selected action in the current state

        if next_state is None:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_state]) # incorporating discounted future reward into the target value

        self.q_table[state][action_idx] += self.alpha * (target - current_q) # updating the Q-value estimate using the Q-learning update rule

    def train(self, num_episodes: int = 10000, return_timing: bool = False):
        """
        Train on random hands.
        Each episode consists of:
        1. draw initial hand,
        2. choose discard action,
        3. draw replacement cards,
        4. observe final score,
        5. update Q-value.

        It also returns timing details for state extraction, action selection, environment simulation, reward calculation, and Q-value updates.
        """
        timing = {"q_training_state_time": 0.0, "q_training_action_selection_time": 0.0, "q_training_environment_time": 0.0, "q_training_reward_time": 0.0, "q_training_update_time": 0.0,}
        
        for _ in range(num_episodes):
            deck = create_deck()
            shuffle_deck(deck)

            initial_hand = draw_cards(deck, self.hand_size)

            start = time.time()
            state = state_features(initial_hand) # converting the drawn hand into a compact feature-based state representation
            timing["q_training_state_time"] += time.time() - start
            
            start = time.time()
            action_idx = self.choose_action(state) # selecting an action index using the epsilon-greedy policy
            keep_indices = self.actions[action_idx]
            timing["q_training_action_selection_time"] += time.time() - start

            start = time.time()
            final_hand = replace_cards(initial_hand, keep_indices, deck) # generating the final hand after replacing discarded cards
            timing["q_training_environment_time"] += time.time() - start

            start = time.time()
            reward = hand_category_score(final_hand, scoring_system=self.scoring_system,) # computing the reward obtained from the resulting final hand
            timing["q_training_reward_time"] += time.time() - start

            start = time.time()
            self.update(state, action_idx, reward) # updating the Q-table using the observed reward from the episode
            timing["q_training_update_time"] += time.time() - start

        if return_timing:
            return timing

    def best_action(self, hand: List[Card], return_timing: bool = False):
        """
        Return the learned best action for a given hand.
        It also returns timing details for state extraction and action lookup.
        """
        timing = {"q_inference_state_time": 0.0, "q_inference_lookup_time": 0.0,}

        start = time.time()
        state = state_features(hand)
        timing["q_inference_state_time"] += time.time() - start

        start = time.time()
        q_values = self.q_table[state] # retrieving the learned Q-values associated with the current state
        best_idx = max(range(len(q_values)), key=lambda i: q_values[i]) # selecting the action index with the highest learned Q-value
        timing["q_inference_lookup_time"] += time.time() - start

        if return_timing:
            return self.actions[best_idx], q_values[best_idx], timing

        return self.actions[best_idx], q_values[best_idx] # returning the best keep/discard action together with its estimated Q-value