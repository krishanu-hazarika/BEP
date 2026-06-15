from collections import Counter
from typing import List
from game import Card

def get_ranks(hand: List[Card]) -> List[int]: # extracting the rank values of all cards in the hand
    return [card.rank for card in hand]

def get_suits(hand: List[Card]) -> List[str]: # extracting the suit values of all cards in the hand
    return [card.suit for card in hand]

def find_same_rank_indices(hand: List[Card], target_count: int) -> List[int]:
    """
    Returns indices of cards whose rank appears exactly target_count times.
    Example:
    - if hand has a pair of 9s and target_count=2, returns indices of those 9s
    """
    ranks = get_ranks(hand)
    rank_counts = Counter(ranks)

    indices = []
    for i, card in enumerate(hand): # iterating through all cards in the hand to identify which card indices belong to ranks appearing target_count times
        if rank_counts[card.rank] == target_count:
            indices.append(i)
    return indices

def find_best_flush_draw(hand: List[Card]) -> List[int]: # identifying the suit with the highest frequency to detect a potential flush draw
    suits = get_suits(hand)
    suit_counts = Counter(suits)

    best_suit, best_count = suit_counts.most_common(1)[0]

    if best_count >= 4: # returning indices of cards belonging to suit of highest frequency if at least four cards match
        return [i for i, card in enumerate(hand) if card.suit == best_suit]

    return []

def is_consecutive(ranks: List[int]) -> bool:
    """
    Checks whether 4 unique ranks form a straight-like sequence.
    Supports A-2-3-4 as low straight possibility.
    """
    unique_ranks = sorted(set(ranks))
    if len(unique_ranks) != 4: # setting up for cases we cannot have a 4-card straight
        return False

    if unique_ranks[-1] - unique_ranks[0] == 3: # setting up for a standard 4-card straight
        return True

    if unique_ranks == [2, 3, 4, 14]: # setting up for a wheel straight possibility (A,2,3,4)
        return True

    return False

def find_best_straight_draw(hand: List[Card]) -> List[int]:
    """
    If there are 4 cards that can form a straight draw, return their indices.
    Otherwise return [].
    """
    ranks = get_ranks(hand)

    for discard_idx in range(len(hand)):
        kept_indices = [i for i in range(len(hand)) if i != discard_idx]
        kept_ranks = [ranks[i] for i in kept_indices]

        if is_consecutive(kept_ranks):
            return kept_indices

    return []

def highest_card_index(hand: List[Card]) -> int:
    """
    Returns the index of the highest-ranked card.
    """
    best_idx = 0
    best_rank = hand[0].rank

    for i, card in enumerate(hand):
        if card.rank > best_rank:
            best_rank = card.rank
            best_idx = i

    return best_idx

def greedy_keep_indices(hand: List[Card]) -> List[int]:
    """
    Simple rule-based heuristic strategy.

    Priority:
    1. Keep 3-of-a-kind
    2. Keep 4-of-a-kind
    3. Keep any pair
    4. Keep 4 to a flush
    5. Keep 4 to a straight
    6. Keep highest card if it is J or higher
    7. Otherwise discard all
    """

    # 1. Keep three-of-a-kind
    triple_indices = find_same_rank_indices(hand, 3)
    if triple_indices:
        return triple_indices

    # 2. Keep four-of-a-kind
    four_indices = find_same_rank_indices(hand, 4)
    if four_indices:
        return four_indices

    # 3. Keep any pair
    pair_indices = find_same_rank_indices(hand, 2)
    if pair_indices:
        return pair_indices

    # 4. Keep 4 cards to a flush
    flush_draw = find_best_flush_draw(hand)
    if flush_draw:
        return flush_draw

    # 5. Keep 4 cards to a straight
    straight_draw = find_best_straight_draw(hand)
    if straight_draw:
        return straight_draw

    # 6. Keep highest card if J or higher
    hi_idx = highest_card_index(hand)
    if hand[hi_idx].rank >= 11:
        return [hi_idx]

    # 7. Otherwise discard everything
    return []