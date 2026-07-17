import itertools
import random
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
import numpy as np
from game import (Card, create_deck, shuffle_deck, draw_cards, replace_cards, hand_category_score, state_features, all_keep_actions,)
import time

NO_DRAW = 0
INSIDE_DRAW = 1      # exactly one rank completes the straight
OPEN_ENDED_DRAW = 2  # two different ranks complete the straight

def _straight_completions(four_ranks: List[int]) -> int:
    """
    Counts how many distinct ranks would turn these 4 unique ranks into a
    5-card straight (counting ace as both high and low).
    """
    rank_set = set(four_ranks)
    completions = 0

    for candidate in range(2, 15):
        if candidate in rank_set:
            continue
        combined = sorted(rank_set | {candidate})
        if combined[-1] - combined[0] == 4 or combined == [2, 3, 4, 5, 14]:
            completions += 1

    return completions

def _best_straight_draw(hand: List[Card]) -> Tuple[Optional[Tuple[int, ...]], int]:
    """
    Looks for the best 4-card straight draw obtainable by discarding
    len(hand) - 4 cards. Returns (exclude_indices, draw_type) where
    exclude_indices are the positions to discard and draw_type is
    OPEN_ENDED_DRAW (two completing ranks), INSIDE_DRAW (one completing
    rank), or NO_DRAW. Prefers the draw with the most completing ranks.
    """
    n = len(hand)
    ranks = [card.rank for card in hand]

    best_exclude = None
    best_completions = 0

    for exclude in itertools.combinations(range(n), n - 4):
        kept_ranks = [ranks[j] for j in range(n) if j not in exclude]
        if len(set(kept_ranks)) != 4:
            continue

        completions = _straight_completions(kept_ranks)
        if completions > best_completions:
            best_completions = completions
            best_exclude = exclude

    if best_exclude is None:
        return None, NO_DRAW

    return best_exclude, OPEN_ENDED_DRAW if best_completions >= 2 else INSIDE_DRAW

def _royal_draw_count(hand: List[Card]) -> Tuple[int, Optional[str]]:
    """
    Returns (count, suit) for the suit holding the most cards of rank 10 or
    higher - the raw material for a royal flush. Counts below 3 are reported
    as (0, None) since they carry no meaningful royal potential.
    """
    royal_counts = Counter(card.suit for card in hand if card.rank >= 10)
    if not royal_counts:
        return 0, None

    suit, count = royal_counts.most_common(1)[0]
    if count < 3:
        return 0, None
    return count, suit

def canonical_hand_order(hand: List[Card]) -> Tuple[List[int], int]:
    """
    Maps a hand onto a canonical card ordering so that a Q-table action index
    (which only refers to a *position*, not a specific card) means the same
    thing for every hand sharing the same state_features() abstraction.

    Without this, the same abstract state (e.g. "one pair, ace high") would be
    dealt to hand positions in random order across episodes, so a fixed action
    index like "keep positions 0,1" would sometimes keep the pair and
    sometimes discard it - the Q-table would just be learning noise.

    Ordering priority:
    1. Cards sharing a rank (pair/trips/quads) are grouped first (highest
       group first).
    2. With a 4+ card flush draw, flush-suited cards are grouped next.
    3. If discarding cards down to 4 leaves a straight draw (open-ended or
       inside), the discarded cards always go last, so "keep everything but
       the tail" reliably means "keep the straight draw".
    4. With 3+ same-suit royal cards (rank 10+), those are grouped first so
       royal-draw keeps are positionally stable.
    Remaining ties break by rank descending.

    Returns (perm, draw_type) where perm[canonical_position] is the index of
    that card in the original hand, and draw_type is NO_DRAW, INSIDE_DRAW,
    or OPEN_ENDED_DRAW.
    """
    n = len(hand)
    rank_counts = Counter(card.rank for card in hand)
    suit_counts = Counter(card.suit for card in hand)
    has_flush_draw = max(suit_counts.values()) >= 4
    all_unique_ranks = all(count == 1 for count in rank_counts.values())

    if all_unique_ranks and not has_flush_draw:
        exclude, draw_type = _best_straight_draw(hand)
        if exclude is not None:
            kept = sorted((i for i in range(n) if i not in exclude), key=lambda i: -hand[i].rank)
            excluded = sorted(exclude, key=lambda i: -hand[i].rank)
            return kept + excluded, draw_type

        royal_count, royal_suit = _royal_draw_count(hand)
        if royal_count >= 3:
            order = sorted(range(n), key=lambda i: (-(hand[i].suit == royal_suit and hand[i].rank >= 10), -hand[i].rank))
            return order, NO_DRAW

    if has_flush_draw:
        order = sorted(range(n), key=lambda i: (-rank_counts[hand[i].rank], -suit_counts[hand[i].suit], -hand[i].rank))
    else:
        order = sorted(range(n), key=lambda i: (-rank_counts[hand[i].rank], -hand[i].rank))

    return order, NO_DRAW

def _max_consecutive_run(ranks: List[int]) -> int:
    """
    Length of the longest run of consecutive unique ranks in the hand,
    counting the ace as both high (14) and low (1).
    """
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    unique = sorted(unique)

    best = 1
    current = 1
    for previous, rank in zip(unique, unique[1:]):
        if rank == previous + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1

    return best

def extended_state_features(canonical_hand: List[Card], draw_type: int) -> tuple:
    """
    Richer state representation for tabular Q-learning, extending
    game.state_features() with draw-oriented information:

    - draw_type: open-ended vs inside straight draw (vs none),
    - run length: longest consecutive-rank run (only 3+ is informative),
    - royal potential: number of same-suit cards of rank 10+ (3+ only),
    - exact high-card composition (which of J/Q/K/A are present), recorded
      only for hands with no pair or better so the state space stays small -
      for made hands the kept group matters far more than the kickers,
    - whether the (highest) pair is Jacks-or-better, for pair/two-pair hands.
    """
    ranks = [card.rank for card in canonical_hand]
    rank_counts = Counter(ranks)
    max_rank_count = max(rank_counts.values())

    run_length = _max_consecutive_run(ranks)
    run_feature = run_length if run_length >= 3 else 0

    royal_count, _ = _royal_draw_count(canonical_hand)

    if max_rank_count == 1:
        high_cards = tuple(sorted((rank for rank in rank_counts if rank >= 11), reverse=True))
    else:
        high_cards = ()

    if max_rank_count == 2:
        best_pair_rank = max(rank for rank, count in rank_counts.items() if count == 2)
        pair_is_high = best_pair_rank >= 11
    else:
        pair_is_high = False

    return state_features(canonical_hand) + (draw_type, run_feature, royal_count, high_cards, pair_is_high)

class QLearningAgent:
    def __init__(self, hand_size: int = 5, scoring_system: str = "video_poker", gamma: float = 0.0, epsilon: float = 0.1,):
        """
        gamma = 0 because each episode is a one-step decision: initial hand -> choose to keep/discard -> final reward.
        epsilon = Exploration probability controlling how often random actions are selected.

        Q-values are updated using an incremental sample average (1/N), which improves convergence for this one-step stationary problem.
        """
        self.hand_size = hand_size
        self.scoring_system = scoring_system
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions: Tuple[Tuple[int, ...], ...] = all_keep_actions(hand_size) # generating all possible keep/discard actions for the selected hand size
        self.q_table: Dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(len(self.actions), dtype=np.float32)) # creating a compact Q-table that maps each state to a vector of estimated action values
        self.visit_counts: Dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(len(self.actions), dtype=np.int64)) # tracking how many times each (state, action) pair has been observed, used to average rewards into the Q-value

    def choose_action(self, state: tuple) -> int:
        """
        Epsilon-greedy action selection.
        Returns the action index.
        """
        # exploring by selecting a random action with probability epsilon
        if random.random() < self.epsilon:
            return random.randrange(len(self.actions))

        q_values = self.q_table[state] # retrieving the estimated Q-values for the current state
        max_q = float(np.max(q_values))
        best_indices = np.flatnonzero(q_values == max_q) # finding all action indices that share the highest Q-value

        return int(best_indices[random.randrange(len(best_indices))]) # randomly selecting one of the best actions to break ties fairly

    def update(self, state: tuple, action_idx: int, reward: float, next_state: tuple = None,) -> None:
        """
        Q-learning update.
        Since this is a one-step episode, next state's future value is usually zero.

        Uses an incremental sample average (learning rate 1/N) rather than a
        fixed rate, so the estimate converges to the true expected reward as
        a state-action pair is visited more often instead of staying noisy.
        """
        current_q = self.q_table[state][action_idx] # retrieving the current Q-value estimate for the selected action in the current state

        if next_state is None:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.q_table[next_state])) # incorporating discounted future reward into the target value

        self.visit_counts[state][action_idx] += 1
        n = self.visit_counts[state][action_idx]

        self.q_table[state][action_idx] += (target - current_q) / n # updating the Q-value estimate as a running average of observed targets

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
            perm, draw_type = canonical_hand_order(initial_hand) # ordering the hand canonically so Q-table action indices mean the same thing across differently-dealt hands
            canonical_hand = [initial_hand[i] for i in perm]
            state = extended_state_features(canonical_hand, draw_type) # converting the drawn hand into a compact feature-based state representation
            timing["q_training_state_time"] += time.time() - start

            start = time.time()
            action_idx = self.choose_action(state) # selecting an action index using the epsilon-greedy policy
            keep_indices = tuple(perm[i] for i in self.actions[action_idx]) # translating the canonical-position action back into real hand indices
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
        perm, draw_type = canonical_hand_order(hand) # same canonical ordering used during training
        canonical_hand = [hand[i] for i in perm]
        state = extended_state_features(canonical_hand, draw_type)
        timing["q_inference_state_time"] += time.time() - start

        start = time.time()
        q_values = self.q_table[state] # retrieving the learned Q-values associated with the current state
        best_idx = int(np.argmax(q_values)) # selecting the action index with the highest learned Q-value
        action = tuple(sorted(perm[i] for i in self.actions[best_idx])) # translating the canonical-position action back into real hand indices, sorted so it is directly comparable with other strategies' actions
        timing["q_inference_lookup_time"] += time.time() - start

        if return_timing:
            return action, q_values[best_idx], timing

        return action, q_values[best_idx] # returning the best keep/discard action together with its estimated Q-value