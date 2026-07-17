import csv
import pickle
from pathlib import Path
from game import Card
from oracle import best_oracle_action, simulate_action
from greedy import greedy_keep_indices
from bandit import best_bandit_action
from qlearning import QLearningAgent
import random
import numpy as np

random.seed(42) # setting a fixed random seed to ensure reproducible and consistent experimental results across multiple runs
np.random.seed(42)

SCORING_SYSTEM = "video_poker"
HAND_SIZE = 5
Q_EPISODES = 100000

ORACLE_SIMULATIONS = 1000
BANDIT_ROUNDS = 300
BANDIT_BUDGET_PER_PULL = 3
BANDIT_EXPLORATION_CONSTANT = 2.0

OUTPUT_FOLDER = Path("case_studies")
OUTPUT_FOLDER.mkdir(exist_ok=True)

POLICY_FOLDER = Path("policies")
POLICY_FOLDER.mkdir(exist_ok=True)
POLICY_FILE = POLICY_FOLDER / f"q_policy_n{HAND_SIZE}_{SCORING_SYSTEM}.pkl"


def card(rank, suit):
    return Card(rank, suit)


def card_list(hand):
    return [str(c) for c in hand]


def action_to_cards(hand, action):
    return [str(hand[i]) for i in action]


def load_or_train_q_agent():
    agent = QLearningAgent(hand_size=HAND_SIZE, scoring_system=SCORING_SYSTEM, gamma=0.0, epsilon=0.1,)

    if POLICY_FILE.exists():
        with open(POLICY_FILE, "rb") as f:
            saved_q_table = pickle.load(f)

        agent.q_table.update(saved_q_table)
        print("Loaded saved Q-learning policy.")

    else:
        print("No saved Q-learning policy found. Training new policy...")
        agent.train(num_episodes=Q_EPISODES)

        with open(POLICY_FILE, "wb") as f:
            pickle.dump(dict(agent.q_table), f)

        print(f"Saved Q-learning policy to {POLICY_FILE}.")

    return agent


def evaluate_case(case_name, hand, q_agent):
    greedy_action = tuple(sorted(greedy_keep_indices(hand))) # normalized to a sorted tuple so it is directly comparable with the oracle's action
    greedy_value = simulate_action(hand, greedy_action, num_simulations=ORACLE_SIMULATIONS, scoring_system=SCORING_SYSTEM,)

    bandit_action, bandit_internal_value = best_bandit_action(hand, num_rounds=BANDIT_ROUNDS, simulation_budget_per_pull=BANDIT_BUDGET_PER_PULL, exploration_constant=BANDIT_EXPLORATION_CONSTANT, scoring_system=SCORING_SYSTEM,)
    bandit_value = simulate_action(hand, bandit_action, num_simulations=ORACLE_SIMULATIONS, scoring_system=SCORING_SYSTEM,)

    q_action, q_internal_value = q_agent.best_action(hand)
    q_value = simulate_action(hand, q_action, num_simulations=ORACLE_SIMULATIONS, scoring_system=SCORING_SYSTEM,)

    oracle_action, oracle_value = best_oracle_action(hand, num_simulations=ORACLE_SIMULATIONS, scoring_system=SCORING_SYSTEM,)

    rows = [
        {
            "case": case_name,
            "initial_hand": " ".join(card_list(hand)),
            "strategy": "Greedy",
            "keep_indices": greedy_action,
            "kept_cards": " ".join(action_to_cards(hand, greedy_action)),
            "estimated_reward": round(greedy_value, 4),
            "matches_oracle": greedy_action == oracle_action,
        },
        {
            "case": case_name,
            "initial_hand": " ".join(card_list(hand)),
            "strategy": "Bandit",
            "keep_indices": bandit_action,
            "kept_cards": " ".join(action_to_cards(hand, bandit_action)),
            "estimated_reward": round(bandit_value, 4),
            "matches_oracle": bandit_action == oracle_action,
        },
        {
            "case": case_name,
            "initial_hand": " ".join(card_list(hand)),
            "strategy": "Q-learning",
            "keep_indices": q_action,
            "kept_cards": " ".join(action_to_cards(hand, q_action)),
            "estimated_reward": round(q_value, 4),
            "matches_oracle": q_action == oracle_action,
        },
        {
            "case": case_name,
            "initial_hand": " ".join(card_list(hand)),
            "strategy": "Oracle",
            "keep_indices": oracle_action,
            "kept_cards": " ".join(action_to_cards(hand, oracle_action)),
            "estimated_reward": round(oracle_value, 4),
            "matches_oracle": True,
        },
    ]

    return rows

def print_case(rows):
    case_name = rows[0]["case"]
    initial_hand = rows[0]["initial_hand"]

    print("=" * 70)
    print(f"Case: {case_name}")
    print(f"Initial hand: {initial_hand}")
    print("-" * 70)

    for row in rows:
        print(f"{row['strategy']}")
        print(f"  Keep indices:      {row['keep_indices']}")
        print(f"  Kept cards:        {row['kept_cards']}")
        print(f"  Estimated reward:  {row['estimated_reward']}")
        print(f"  Matches oracle:    {row['matches_oracle']}")
        print()

    print()


def save_rows_to_csv(all_rows):
    output_file = OUTPUT_FOLDER / f"case_studies_{SCORING_SYSTEM}.csv"

    fieldnames = ["case", "initial_hand", "strategy", "keep_indices", "kept_cards", "estimated_reward", "matches_oracle",]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved case study results to {output_file}")


def main():
    q_agent = load_or_train_q_agent()

    test_hands = {
        "Pair hand": [card(14, "H"), card(14, "D"), card(7, "C"), card(5, "S"), card(3, "D")],
        "Four-to-a-flush": [card(14, "H"), card(13, "H"), card(12, "H"), card(7, "H"), card(3, "D")],
        "Four-to-a-straight": [card(5, "H"), card(6, "D"), card(7, "C"), card(8, "S"), card(13, "H")],
        "Two pair": [card(14, "H"), card(14, "D"), card(13, "C"), card(13, "D"), card(3, "S")],
        "Difficult high-card hand": [card(14, "H"), card(13, "D"), card(12, "C"), card(11, "S"), card(4, "D")],
    }

    all_rows = []

    for case_name, hand in test_hands.items():
        rows = evaluate_case(case_name, hand, q_agent)
        print_case(rows)
        all_rows.extend(rows)

    save_rows_to_csv(all_rows)


if __name__ == "__main__":
    main()