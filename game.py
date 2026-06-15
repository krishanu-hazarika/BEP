from dataclasses import dataclass
from collections import Counter
import random
from typing import List, Tuple
import itertools

SUITS = ["H", "D", "C", "S"] # hearts, diamonds, clubs, spades
RANKS = list(range(2, 15))  # special cards are 11(jack), 12(queen), 13(king), 14(ace)

# representing a playing card as an immutable object with a rank and suit
@dataclass(frozen=True)
class Card:
    rank: int
    suit: str

    def __str__(self) -> str: # defining how a card is displayed as a readable string (for example, we display "AH" for Ace of Hearts)
        rank_map = {11: "J", 12: "Q", 13: "K", 14: "A"}
        rank_str = rank_map.get(self.rank, str(self.rank))
        return f"{rank_str}{self.suit}"

LINEAR_SCORING = { 
    "High Card": 0,
    "One Pair": 1,
    "Two Pair": 2,
    "Three of a Kind": 3,
    "Straight": 4,
    "Flush": 5,
    "Full House": 6,
    "Four of a Kind": 7,
    "Straight Flush": 8,
    "Royal Flush": 9,
}

VIDEO_POKER_SCORING = { # scoring based on video poker payout mentioned on Ethier (2016)
    "High Card": 0,
    "One Pair": 1,
    "Two Pair": 2,
    "Three of a Kind": 3,
    "Straight": 4,
    "Flush": 6,
    "Full House": 9,
    "Four of a Kind": 25,
    "Straight Flush": 50,
    "Royal Flush": 800,
}

def create_deck() -> List[Card]: # creating a standard 52-card deck containing all rank and suit combinations
    return [Card(rank, suit) for suit in SUITS for rank in RANKS]

def shuffle_deck(deck: List[Card]) -> None: # shuffling the 52-card deck randomly
    random.shuffle(deck)

def draw_cards(deck: List[Card], n: int) -> List[Card]: # drawing 'n' cards from the deck and removing them from future play
    if n > len(deck):
        raise ValueError("Not enough cards left in deck to draw.")
    drawn = deck[:n]
    del deck[:n]
    return drawn

def replace_cards(hand: List[Card], keep_indices: List[int], deck: List[Card]) -> List[Card]: # replacing discarded cards while preserving selected cards in the hand
    """
    keep_indices: indices (0-based) of cards to keep from the current hand.
    All other cards are discarded and replaced from the deck.
    """
    keep_set = set(keep_indices)

    if any(i < 0 or i >= len(hand) for i in keep_indices):
        raise ValueError("keep_indices contains invalid hand index.")

    new_hand = []
    cards_to_draw = 0

    for i, card in enumerate(hand):
        if i in keep_set:
            new_hand.append(card)
        else:
            cards_to_draw += 1

    new_hand.extend(draw_cards(deck, cards_to_draw))
    return new_hand

def is_straight(ranks: List[int]) -> bool:
    """
    Checks for straights including the special A-2-3-4-5 case.
    """
    unique_ranks = sorted(set(ranks)) # checking if all cards have different ranks, otherwise a straight is not possible
    if len(unique_ranks) != 5:
        return False

    # checking if the ranks are consecutive (standard straight)
    if unique_ranks[-1] - unique_ranks[0] == 4:
        return True

    # checking for special case straight which is wheel straight (A,2,3,4,5)
    if unique_ranks == [2, 3, 4, 5, 14]:
        return True

    return False

def evaluate_5_card_hand(hand: List[Card]) -> Tuple[int, str]:
    """
    Returns (score, hand_name).
    
    Higher score indicates a stronger poker hand and is used only for internal hand ranking.
    
    Final experimental rewards are assigned separately through the selected scoring system.
    """
    if len(hand) != 5:
        raise ValueError("Hand evaluation currently expects exactly 5 cards.")

    ranks = sorted([card.rank for card in hand]) # extracting all card ranks from the hand and sorting them in an ascending order
    suits = [card.suit for card in hand] # extracting the suit of each card in the hand

    rank_counts = Counter(ranks) # counting the number of times each card rank appears in the hand
    count_values = sorted(rank_counts.values(), reverse=True) # creating a descending list of rank frequencies

    flush = len(set(suits)) == 1 # checking if all cards have the same suit, if that is the case then it is a flush
    straight = is_straight(ranks) # checking whether ranks form a straight sequence

    if flush and sorted(ranks) == [10, 11, 12, 13, 14]:
        return 9, "Royal Flush"
    if flush and straight:
        return 8, "Straight Flush"
    if count_values == [4, 1]:
        return 7, "Four of a Kind"
    if count_values == [3, 2]:
        return 6, "Full House"
    if flush:
        return 5, "Flush"
    if straight:
        return 4, "Straight"
    if count_values == [3, 1, 1]:
        return 3, "Three of a Kind"
    if count_values == [2, 2, 1]:
        return 2, "Two Pair"
    if count_values == [2, 1, 1, 1]:
        return 1, "One Pair"
    return 0, "High Card"

def evaluate_hand(hand: List[Card], scoring_system: str = "linear") -> Tuple[int, str]:
    """
    Evaluates a hand of size n >= 5 by selecting the best 5-card subset.

    scoring_system:
    - "linear": scores from 0 to 9
    - "video_poker": payout-style scoring
    """
    if len(hand) < 5:
        raise ValueError("Hand must contain at least 5 cards.")

    if scoring_system == "linear":
        scoring_table = LINEAR_SCORING
    elif scoring_system == "video_poker":
        scoring_table = VIDEO_POKER_SCORING
    else:
        raise ValueError("Unknown scoring system.")

    best_score = float("-inf")
    best_name = None

    for five_card_subset in itertools.combinations(hand, 5): # iterating through every possible 5-card combination from the hand to determine the highest-scoring poker hand
        _, hand_name = evaluate_5_card_hand(list(five_card_subset))
        score = scoring_table[hand_name]

        if score > best_score:
            best_score = score
            best_name = hand_name

    return best_score, best_name

def play_round(keep_indices: List[int], hand_size: int = 5, scoring_system: str = "linear") -> Tuple[List[Card], List[Card], Tuple[int, str]]: # simulating a complete round of draw poker by creating a deck, drawing an initial hand, replacing discarded cards, and evaluating the final hand

    deck = create_deck()
    shuffle_deck(deck)

    initial_hand = draw_cards(deck, hand_size)
    final_hand = replace_cards(initial_hand, keep_indices, deck)
    result = evaluate_hand(final_hand, scoring_system=scoring_system)

    return initial_hand, final_hand, result

def all_keep_actions(hand_size: int = 5) -> List[List[int]]: 
    """
    Returns all possible keep actions as lists of indices.
    For hand_size = 5, this gives 2^5 = 32 possible actions.
    Example actions:
    []            -> discard all cards
    [0, 1]        -> keep cards at indices 0 and 1
    [0, 1, 2, 3]  -> keep first four cards
    [0, 1, 2, 3, 4] -> keep all cards
    """
    actions = []
    indices = list(range(hand_size))

    for r in range(hand_size + 1):
        for combo in itertools.combinations(indices, r):
            actions.append(list(combo))

    return actions

def hand_category_score(hand: List[Card], scoring_system: str = "linear") -> int: # returning the numerical score of a hand under the selected scoring system (either linear or video poker scoring system)
    score, _ = evaluate_hand(hand, scoring_system=scoring_system)
    return score

def state_features(hand: List[Card]) -> tuple:
    """
    Compact state representation for Q-learning.

    Features:
    - sorted rank counts (e.g. pair -> (2,1,1,1), trips -> (3,1,1))
    - max suit count (for flush potential)
    - number of unique ranks
    - highest rank in hand

    This is intentionally compact, so states are shared across similar hands.
    """
    ranks = [card.rank for card in hand]
    suits = [card.suit for card in hand]

    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    count_pattern = tuple(sorted(rank_counts.values(), reverse=True)) # creating a descending frequency pattern of rank occurrences
    max_suit_count = max(suit_counts.values())
    num_unique_ranks = len(rank_counts)
    highest_rank = max(ranks)

    return (count_pattern, max_suit_count, num_unique_ranks, highest_rank)