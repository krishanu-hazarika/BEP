from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

EVAL_FOLDER = Path("evaluation")
PLOT_FOLDER = Path("plots")
PLOT_FOLDER.mkdir(exist_ok=True)

def load_results():
    """
    Loads all experimental evaluation CSV files from the evaluation folder
    and combines them into a single pandas DataFrame.

    The function searches for all CSV files inside the evaluation directory,
    reads each file individually, and concatenates them into one unified
    table for further analysis and visualization.

    Raises:
        FileNotFoundError:
            If no evaluation CSV files are found in the folder.

    Returns:
        pd.DataFrame:
            Combined DataFrame containing all experimental results.
    """
    csv_files = list(EVAL_FOLDER.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in evaluation folder.")

    dfs = [pd.read_csv(file) for file in csv_files]
    return pd.concat(dfs, ignore_index=True)

def plot_average_score_no_ci(df):
    """
    Plots the average expected reward achieved by each strategy across
    different hand sizes and scoring systems.

    The function generates line plots for greedy search, bandit search,
    Q-learning, and oracle search using the recorded average rewards
    from the experimental results DataFrame. Separate curves are shown
    for each scoring system.

    The resulting figure is saved as:
    average_score_no_CI.png

    Parameters:
        df (pd.DataFrame):
            DataFrame containing aggregated experimental evaluation results.
    """
    plt.figure(figsize=(8, 5))

    for scoring_system in df["scoring_system"].unique():
        subset = df[df["scoring_system"] == scoring_system]

        for strategy in ["greedy", "bandit", "qlearning", "oracle"]:
            plt.plot(subset["hand_size"], subset[f"avg_{strategy}"], marker="o", label=f"{strategy} ({scoring_system})",)

    plt.xlabel("Hand size (n)")
    plt.ylabel("Average expected score")
    plt.title("Average expected score")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FOLDER / "average_score_no_CI.png", dpi=300)
    plt.close()

def plot_match_rate(df):
    """
    Plots the action agreement rate between each strategy and the oracle
    across different hand sizes and scoring systems.

    The function generates line plots for greedy search, bandit search,
    and Q-learning using the recorded oracle match rates from the
    experimental results DataFrame. Higher agreement rates indicate
    that a strategy selected the same keep/discard action as the oracle
    more frequently.

    The resulting figure is saved as:
    match_rate.png

    Parameters:
        df (pd.DataFrame):
            DataFrame containing aggregated experimental evaluation results.
    """
    plt.figure(figsize=(8, 5))

    for scoring_system in df["scoring_system"].unique():
        subset = df[df["scoring_system"] == scoring_system]

        for strategy in ["greedy", "bandit", "qlearning"]:
            plt.plot(subset["hand_size"], subset[f"{strategy}_match_rate"], marker="o", label=f"{strategy} ({scoring_system})",)

    plt.xlabel("Hand size (n)")
    plt.ylabel("Same-action rate vs oracle")
    plt.title("Action agreement with oracle")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FOLDER / "match_rate.png", dpi=300)
    plt.close()

def plot_strategy_runtime(df):
    """
    Plots runtime of each strategy separately.
    """
    plt.figure(figsize=(8, 5))

    for scoring_system in df["scoring_system"].unique():
        subset = df[df["scoring_system"] == scoring_system]

        for strategy in ["greedy", "bandit", "qlearning", "oracle"]:
            plt.plot(subset["hand_size"], subset[f"{strategy}_runtime_seconds"], marker="o", label=f"{strategy} ({scoring_system})",)

    plt.xlabel("Hand size (n)")
    plt.ylabel("Runtime in log scale (seconds)")
    plt.yscale("log")
    plt.title("Runtime by strategy")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FOLDER / "runtime_by_strategy.png", dpi=300)
    plt.close()

def plot_runtime_decomposition(df):
    """
    Plots internal runtime breakdown.
    """

    components = {
        "Bandit action generation": df["bandit_action_generation_time"].sum(),
        "Bandit UCB selection": df["bandit_ucb_selection_time"].sum(),
        "Bandit simulation": df["bandit_simulation_time"].sum(),
        "Bandit update": df["bandit_update_time"].sum(),
        "Oracle action generation": df["oracle_action_generation_time"].sum(),
        "Oracle simulation": df["oracle_simulation_time"].sum(),
        "Q state extraction": df["q_training_state_time"].sum(),
        "Q action selection": df["q_training_action_selection_time"].sum(),
        "Q environment": df["q_training_environment_time"].sum(),
        "Q reward calculation": df["q_training_reward_time"].sum(),
        "Q update": df["q_training_update_time"].sum(),
    }

    plt.figure(figsize=(10, 6))
    plt.barh(list(components.keys()), list(components.values()))
    plt.xlabel("Runtime in log scale (seconds)")
    plt.xscale("log")
    plt.title("Runtime decomposition")
    plt.tight_layout()
    plt.savefig(PLOT_FOLDER / "runtime_decomposition.png", dpi=300)
    plt.close()

def plot_risk_reward(df):
    """
    Plots the relationship between reward variability and average expected reward
    for all strategies under different scoring systems.

    The function generates scatter plots using the standard deviation of reward
    as a measure of risk and the average expected reward as a measure of performance.
    Each point corresponds to one experimental configuration and is annotated
    with its hand size.

    Different marker styles are used to distinguish between linear scoring
    and video-poker scoring systems.

    The resulting figure is saved as:
    risk_reward.png

    Parameters:
        df (pd.DataFrame):
            DataFrame containing aggregated experimental evaluation results.
    """
    plt.figure(figsize=(8, 6))

    strategies = ["greedy", "bandit", "qlearning", "oracle"]
    markers = {"linear": "o", "video_poker": "s",}

    for scoring_system in df["scoring_system"].unique():
        subset = df[df["scoring_system"] == scoring_system]

        for strategy in strategies:
            plt.scatter(subset[f"std_{strategy}"], subset[f"avg_{strategy}"], marker=markers.get(scoring_system, "o"), s=100, label=f"{strategy} ({scoring_system})",)

            for _, row in subset.iterrows():
                plt.annotate(
                    f"n={int(row['hand_size'])}", (row[f"std_{strategy}"], row[f"avg_{strategy}"],),fontsize=8,)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    plt.legend(unique.values(), unique.keys(), fontsize=8)
    plt.xlabel("Standard deviation of reward")
    plt.ylabel("Average expected reward")
    plt.title("Risk-reward tradeoff")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FOLDER / "risk_reward.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    df = load_results()

    plot_average_score_no_ci(df)
    plot_match_rate(df)
    plot_strategy_runtime(df)
    plot_runtime_decomposition(df)
    plot_risk_reward(df)
    
    print("Plots saved to plots folder.")