# Draw Poker Strategy Optimization

## Overview

This project compares four decision-making strategies for draw poker, each with a different balance of computational cost and decision quality. The objective is to determine which cards should be kept or discarded in order to maximize the expected reward of the final hand.

| Strategy   | Approach                      | Role                          |
| ---------- | ----------------------------- | ----------------------------- |
| Greedy     | Rule-based heuristics         | Fast baseline                 |
| Oracle     | Exhaustive Monte Carlo search | Approximate benchmark         |
| Bandit     | UCB multi-armed bandit        | Efficient near-optimal search |
| Q-Learning | Reinforcement learning agent  | Learned policy                |

The project was developed as part of a Bachelor End Project in Data Science and investigates how strategy performance changes under increasing action-space complexity and alternative reward structures.

---

## Project Structure

```text
project_root/
│
├── game.py              # Core poker environment & hand evaluation
├── greedy.py            # Rule-based heuristic strategy
├── oracle.py            # Exhaustive Monte Carlo search
├── bandit.py            # UCB multi-armed bandit search
├── qlearning.py         # Q-learning reinforcement learning agent
├── evaluate.py          # Experimental evaluation pipeline
├── plots.py             # Figure generation
├── demo.py              # Interactive strategy demonstration
├── case_studies.py      # Representative hand analysis
├── requirements.txt
├── README.md
│
├── evaluation/
│   ├── evaluation_n5_linear.csv
│   ├── evaluation_n6_linear.csv
│   ├── evaluation_n7_linear.csv
│   ├── evaluation_n5_video_poker.csv
│   ├── evaluation_n6_video_poker.csv
│   └── evaluation_n7_video_poker.csv
│
├── plots/
│   ├── average_score_no_CI.png
│   ├── match_rate.png
│   ├── runtime_by_strategy.png
│   ├── runtime_decomposition.png
│   └── risk_reward.png
│
├── policies/
│   └── q_policy_*.pkl
│
└── case_studies/
    └── case_studies_video_poker.csv
```

---

## Module Reference

### game.py — Poker Engine

Core environment and game mechanics:

* Card representation
* Deck generation and shuffling
* Draw and replacement mechanics
* Poker hand evaluation
* Scoring systems
* State feature extraction for reinforcement learning
* Enumeration of all keep/discard actions

### greedy.py — Heuristic Strategy

Fast rule-based baseline using poker-specific heuristics such as pairs, flush draws, straight draws, and high-card retention.

### oracle.py — Exhaustive Monte Carlo Search

Evaluates every possible keep/discard action using Monte Carlo simulation and selects the action with the highest estimated expected reward.

Serves as the approximate benchmark used throughout the experiments.

### bandit.py — UCB Bandit Search

Treats each keep/discard action as a bandit arm and uses the Upper Confidence Bound (UCB) algorithm to balance exploration and exploitation.

Attempts to approximate oracle performance while using substantially fewer simulations.

### qlearning.py — Reinforcement Learning Agent

Learns action values through repeated simulated play using tabular Q-learning and ε-greedy exploration.

The discard problem is modeled as a one-step episodic reinforcement learning task.

### evaluate.py — Experimental Evaluation Pipeline

Trains Q-learning agents and evaluates all strategies on identical randomly generated poker hands.

Records:

* Average expected reward
* Reward variability (standard deviation)
* Regret relative to oracle
* Oracle agreement rates
* Statistical significance tests
* Runtime measurements
* Runtime decomposition metrics

Results are saved as CSV files.

### plots.py — Visualization

Loads evaluation results and generates all figures used in the analysis.

### demo.py — Interactive Demonstration

Generates a random poker hand and shows:

* Strategy recommendations
* Kept and discarded cards
* Estimated rewards
* Realized outcomes after replacement

Useful for understanding individual strategy behavior.

### case_studies.py — Case Study Generation

Evaluates predefined representative poker hands and records:

* Selected actions
* Estimated rewards
* Oracle agreement

Results are exported to CSV for inclusion in qualitative analysis.

---

## Installation

### 1. Clone the repository

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

* numpy
* pandas
* matplotlib
* scipy

Standard library modules used:

* random
* itertools
* collections
* csv
* pathlib
* time
* math
* pickle

---

## Usage

### Run Full Experimental Evaluation

```bash
python evaluate.py
```

This will:

* Train Q-learning agents
* Evaluate all strategies
* Generate evaluation CSV files

Results are stored in:

```text
evaluation/
```

---

### Generate Plots

```bash
python plots.py
```

Generated figures are saved to:

```text
plots/
```

---

### Run Interactive Demo

```bash
python demo.py
```

Displays a randomly generated poker hand together with recommendations from all strategies and a realized outcome.

---

### Generate Case Studies

```bash
python case_studies.py
```

Produces detailed strategy comparisons on representative hands and saves results to:

```text
case_studies/
```

---

## Generated Outputs

### Evaluation Results

```text
evaluation_n5_linear.csv
evaluation_n6_linear.csv
evaluation_n7_linear.csv
evaluation_n5_video_poker.csv
evaluation_n6_video_poker.csv
evaluation_n7_video_poker.csv
```

Each file contains:

* Experimental configuration
* Average rewards
* Standard deviations
* Regret values
* Oracle agreement rates
* Statistical significance test results
* Runtime statistics
* Runtime decomposition metrics

---

### Figures

| File                      | Contents                           |
| ------------------------- | ---------------------------------- |
| average_score_no_CI.png   | Average expected reward            |
| match_rate.png            | Action agreement with oracle       |
| runtime_by_strategy.png   | Runtime scaling by strategy        |
| runtime_decomposition.png | Internal runtime breakdown         |
| risk_reward.png           | Reward versus variability analysis |

---

## Key Findings

* Oracle search consistently achieves the highest expected reward across all hand sizes and scoring systems.
* Bandit search closely approximates oracle performance while requiring substantially less computation.
* Greedy heuristics perform competitively despite negligible computational cost.
* Q-learning struggles to generalize effectively using the compact state representation employed in this project.
* Performance differences become more pronounced as hand size increases.
* Video-poker scoring amplifies reward variability and increases the importance of accurate discard decisions.
* Monte Carlo simulation dominates overall computational cost for search-based methods.

---

## Reproducibility

All experiments use fixed random seeds:

```python
random.seed(42)
numpy.random.seed(42)
```

This ensures consistent experimental results across repeated executions.
