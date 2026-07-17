# Draw Poker Strategy Optimization

## Overview

This project investigates decision-making strategies for Draw Poker by comparing four approaches that differ in computational cost and decision quality. The objective is to determine which cards should be kept or discarded in order to maximize the expected reward of the final poker hand.

| Strategy | Approach | Role |
|----------|----------|------|
| Greedy | Rule-based heuristics | Fast baseline |
| Oracle | Exhaustive Monte Carlo search | Approximate benchmark |
| Bandit | Upper Confidence Bound (UCB) search | Efficient near-optimal search |
| Q-Learning | Enhanced tabular reinforcement learning | Learned policy |

The project was developed as part of a Bachelor End Project in Data Science and evaluates how strategy performance changes under increasing action-space complexity in a Video Poker scoring system.

---

# Project Structure

```text
project_root/
│
├── game.py              # Core poker environment and hand evaluation
├── greedy.py            # Rule-based heuristic strategy
├── oracle.py            # Exhaustive Monte Carlo search
├── bandit.py            # UCB multi-armed bandit search
├── qlearning.py         # Enhanced tabular Q-learning agent
├── evaluate.py          # Experimental evaluation pipeline
├── plots.py             # Figure generation
├── demo.py              # Interactive strategy demonstration
├── case_studies.py      # Representative hand analysis
├── requirements.txt
├── README.md
│
├── evaluation/
│   ├── evaluation_n5_video_poker.csv
│   ├── evaluation_n6_video_poker.csv
│   └── evaluation_n7_video_poker.csv
│
├── plots/
│   ├── average_score_no_CI.png
│   ├── match_rate.png
│   ├── runtime_by_strategy.png
│   └── runtime_decomposition.png
│
├── policies/
│   └── q_policy_*.pkl
│
└── case_studies/
    └── case_studies_video_poker.csv
```

---

# Module Reference

## game.py — Poker Engine

Provides the core game environment, including:

- Card representation
- Deck generation and shuffling
- Drawing and replacing cards
- Poker hand evaluation
- Video Poker scoring
- State feature extraction
- Enumeration of all possible keep/discard actions

---

## greedy.py — Heuristic Strategy

Implements a rule-based baseline using poker-specific heuristics such as:

- Pairs
- Flush draws
- Straight draws
- High-card retention

The strategy is computationally inexpensive and provides a strong baseline for comparison.

---

## oracle.py — Exhaustive Monte Carlo Search

Evaluates every possible keep/discard action using Monte Carlo simulation and selects the action with the highest estimated expected reward.

Although computationally expensive, it serves as the approximate benchmark throughout the project.

---

## bandit.py — UCB Bandit Search

Treats each keep/discard action as an independent bandit arm and applies the Upper Confidence Bound (UCB) algorithm to balance exploration and exploitation.

The method approximates Oracle performance while requiring substantially fewer simulations.

---

## qlearning.py — Enhanced Tabular Q-Learning

Implements an enhanced tabular Q-learning agent for the one-step discard decision problem.

Key improvements include:

- Canonical hand ordering to align actions with state representations.
- Differentiation between inside and open-ended straight draws.
- Longest consecutive rank run feature.
- Royal flush potential feature.
- Exact high-card composition for high-card hands.
- Jacks-or-Better pair indicator.
- Incremental sample-average Q-value updates for stable convergence.

The discard decision is modelled as a one-step reinforcement learning problem where each episode consists of:

1. Drawing an initial hand.
2. Selecting a keep/discard action.
3. Drawing replacement cards.
4. Receiving the final reward.
5. Updating the Q-table.

---

## evaluate.py — Experimental Evaluation

Automatically trains a Q-learning agent and evaluates all four strategies on identical randomly generated poker hands.

For each strategy it records:

- Average expected reward
- Reward standard deviation
- Regret relative to Oracle
- Oracle agreement rate
- Paired t-test significance values
- Strategy runtime
- Runtime decomposition
- Q-learning training and inference timings

Results are exported as CSV files.

---

## plots.py — Visualization

Reads evaluation results and generates all figures used throughout the analysis, including:

- Average expected reward
- Oracle agreement rate
- Runtime comparison
- Runtime decomposition

---

## demo.py — Demonstration

Generates a random poker hand and displays:

- Strategy recommendations
- Cards kept and discarded
- Estimated rewards
- Final realized hand
- Final reward

Useful for understanding the behaviour of individual strategies.

---

## case_studies.py — Representative Hands

Evaluates a number of predefined poker hands and reports:

- Selected actions
- Cards retained
- Estimated expected reward
- Agreement with Oracle

Results are exported as CSV files for qualitative analysis.

---

# Installation

## Clone the repository

```bash
git clone <repository_url>
cd <repository_name>
```

## Install dependencies

```bash
pip install -r requirements.txt
```

Required packages:

- numpy
- matplotlib
- pandas
- scipy

The project additionally uses several Python standard-library modules, including:

- random
- itertools
- collections
- csv
- pathlib
- pickle
- time

---

# Usage

## Run the Experimental Evaluation

```bash
python evaluate.py
```

This will:

- Train the Q-learning agent.
- Evaluate all four strategies.
- Save evaluation results as CSV files.

Output:

```text
evaluation/
```

---

## Generate Figures

```bash
python plots.py
```

Output:

```text
plots/
```

---

## Run the Demonstration

```bash
python demo.py
```

Displays a randomly generated poker hand together with recommendations from all strategies and the realized outcome after replacement.

---

## Generate Case Studies

```bash
python case_studies.py
```

Produces detailed comparisons for representative poker hands.

Output:

```text
case_studies/
```

---

# Generated Outputs

## Evaluation Results

```text
evaluation_n5_video_poker.csv
evaluation_n6_video_poker.csv
evaluation_n7_video_poker.csv
```

Each evaluation file contains:

- Experimental configuration
- Average expected reward
- Standard deviation
- Regret
- Oracle agreement rate
- Paired t-test results
- Runtime statistics
- Runtime decomposition
- Q-learning training and inference timings

---

## Figures

| File | Description |
|------|-------------|
| average_score_no_CI.png | Average expected reward across strategies |
| match_rate.png | Action agreement with Oracle |
| runtime_by_strategy.png | Runtime comparison between strategies |
| runtime_decomposition.png | Internal runtime breakdown |

---

# Key Findings

- Oracle search consistently achieves the highest expected reward across all evaluated hand sizes.
- Bandit search closely approximates Oracle performance while requiring substantially less computation.
- Greedy heuristics provide competitive performance despite their negligible computational cost.
- Although Q-learning does not outperform the search-based methods, it achieves competitive expected rewards while maintaining low inference cost.
- Performance differences become more pronounced as the hand size increases.
- Monte Carlo simulation dominates the computational cost of the search-based strategies.

---

# Reproducibility

All experiments use fixed random seeds:

```python
random.seed(42)
np.random.seed(42)
```

ensuring reproducible experimental results across repeated executions.

---