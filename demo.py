import random
import pickle
from pathlib import Path
from game import create_deck, shuffle_deck, draw_cards, replace_cards, evaluate_hand
from greedy import greedy_keep_indices
from bandit import best_bandit_action
from oracle import best_oracle_action, simulate_action
from qlearning import QLearningAgent

# random.seed(42)

HAND_SIZE = 7
SCORING_SYSTEM = "video_poker"
Q_EPISODES = 20000

POLICY_FOLDER = Path("policies")
POLICY_FOLDER.mkdir(exist_ok=True)

POLICY_FILE = POLICY_FOLDER / f"q_policy_n{HAND_SIZE}_{SCORING_SYSTEM}.pkl"


def card_list(cards):
    return [str(card) for card in cards]


def keep_cards(hand, keep_indices):
    return [str(hand[i]) for i in keep_indices]

def discard_cards(hand, keep_indices):
    keep_set = set(keep_indices)
    return [str(card) for i, card in enumerate(hand) if i not in keep_set]

def play_realized_round(initial_hand, keep_indices):
    """
    Plays one concrete realization after a strategy chooses which cards to keep.
    This shows the actual discarded cards, replacement cards, final hand,
    final hand category, and final score.
    """
    deck = create_deck()
    for card in initial_hand:
        deck.remove(card)
    shuffle_deck(deck)
    cards_to_replace = len(initial_hand) - len(keep_indices)
    replacement_cards = deck[:cards_to_replace]
    final_hand = replace_cards(initial_hand, keep_indices, deck)
    final_score, final_name = evaluate_hand(
        final_hand,
        scoring_system=SCORING_SYSTEM,
    )
    return replacement_cards, final_hand, final_score, final_name

def load_or_train_q_agent():
    agent = QLearningAgent(
        hand_size=HAND_SIZE,
        scoring_system=SCORING_SYSTEM,
        alpha=0.1,
        gamma=0.0,
        epsilon=0.1,
    )

    if POLICY_FILE.exists():
        with open(POLICY_FILE, "rb") as f:
            saved_q_table = pickle.load(f)

        agent.q_table.update(saved_q_table)
        print("Loaded saved Q-learning policy.\n")

    else:
        print("No saved Q-learning policy found.")
        print("Training Q-learning agent once and saving policy...\n")

        agent.train(num_episodes=Q_EPISODES)

        with open(POLICY_FILE, "wb") as f:
            pickle.dump(dict(agent.q_table), f)

        print(f"Saved Q-learning policy to {POLICY_FILE}.\n")

    return agent

		
def print_strategy_result(strategy_name, initial_hand, action, estimated_value, learned_q_value=None):
    replacement_cards, final_hand, final_score, final_name = play_realized_round(
        initial_hand,
        action,
    )
    print(strategy_name)
    print("Keep indices:      ", action)
    print("Kept cards:        ", keep_cards(initial_hand, action))
    print("Discarded cards:   ", discard_cards(initial_hand, action))
    print("Replacement cards: ", card_list(replacement_cards))
    print("Final hand:        ", card_list(final_hand))
    if learned_q_value is not None:
        print("Learned Q-value:   ", round(learned_q_value, 4))
    print("Estimated reward:  ", round(estimated_value, 4))
    print("Realized result:   ", final_name)
    print("Realized score:    ", final_score)
    print()

def run_demo():
    print("=" * 70)
    print("DRAW POKER STRATEGY DEMO")
    print("=" * 70)

    print(f"Hand size: {HAND_SIZE}")
    print(f"Scoring system: {SCORING_SYSTEM}")
    print()

    q_agent = load_or_train_q_agent()

    deck = create_deck()
    shuffle_deck(deck)
    initial_hand = draw_cards(deck, HAND_SIZE)

    print("Initial hand:")
    print(card_list(initial_hand))
    initial_score, initial_name = evaluate_hand(initial_hand, scoring_system=SCORING_SYSTEM,)
    print("Initial result:", initial_name)
    print("Initial score: ", initial_score)
    print()

    print("-" * 70)
    print("Strategy recommendations and realized outcomes")
    print("-" * 70)

    greedy_action = greedy_keep_indices(initial_hand)
    greedy_value = simulate_action(initial_hand, greedy_action, num_simulations=300, scoring_system=SCORING_SYSTEM,)
    print_strategy_result("Greedy strategy", initial_hand, greedy_action, greedy_value,)

    bandit_action, _ = best_bandit_action(initial_hand, num_rounds=100, simulation_budget_per_pull=3, exploration_constant=2.0, scoring_system=SCORING_SYSTEM,)
    bandit_value = simulate_action(initial_hand, bandit_action, num_simulations=300, scoring_system=SCORING_SYSTEM,)
    print_strategy_result("Bandit strategy", initial_hand, bandit_action, bandit_value,)

    q_action, q_estimated_value = q_agent.best_action(initial_hand)
    q_value = simulate_action(initial_hand, q_action, num_simulations=300, scoring_system=SCORING_SYSTEM,)
    print_strategy_result("Q-learning strategy", initial_hand, q_action, q_value, learned_q_value=q_estimated_value,)

    oracle_action, oracle_value = best_oracle_action(initial_hand, num_simulations=300, scoring_system=SCORING_SYSTEM,)
    print_strategy_result("Oracle strategy", initial_hand, oracle_action, oracle_value,)

    print("=" * 70)
    print("Demo completed.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()