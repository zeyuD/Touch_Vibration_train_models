import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from utils.aggregate_vote import aggregate_predictions

from functions.load_machine_config import load_machine_config

# Simulate or load real saved prediction scores and labels
def load_saved_scores_and_labels(result_dir, target_key):
    """
    Simulate loading previously saved predictions and labels.
    Replace with loading actual .npy or .csv if available.
    """
    pred_path = os.path.join(result_dir, target_key, "raw_scores.npy")
    label_path = os.path.join(result_dir, target_key, "raw_labels.npy")
    scores = np.load(pred_path)
    labels = np.load(label_path)
    # print("Scores:", scores, "Shape:", scores.shape)
    # print("Labels:", labels, "Shape:", labels.shape)
    return scores, labels

def plot_vote_result(metrics, save_path, title):
    plt.figure(figsize=(8, 6))
    df_cm = pd.DataFrame(metrics["conf_matrix"], index=["Impostor", "Genuine"], columns=["Impostor", "Genuine"])
    sn.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{title} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    config = load_machine_config()
    work_dir = os.path.join(config["data_dir"], "Touch_Vibration")
    result_dir = os.path.join(work_dir, "verification_results_zscore")

    usernames = ["crystal", "james", "kevin", "will"]
    fingernames = ["0", "1", "2", "3"]

    all_metrics = []

    group_size = 5
    threshold = 0.5

    for username in usernames:
        for finger in fingernames:
            target_key = f"{username}_{finger}"
            try:
                scores, labels = load_saved_scores_and_labels(result_dir, target_key)
                metrics = aggregate_predictions(scores, labels, group_size=group_size, threshold=threshold)
                all_metrics.append(metrics)

                print(f"{target_key}: Accuracy={metrics['accuracy']:.2f}%, AUC={metrics['auc']:.4f}, FAR={metrics['far']:.4f}, FRR={metrics['frr']:.4f}")

                # Save plot
                save_dir = os.path.join(result_dir, target_key)
                plot_vote_result(metrics, os.path.join(save_dir, "vote5_conf_matrix.png"), target_key)

            except Exception as e:
                print(f"Failed to process {target_key}: {e}")

    # Aggregate across users
    if all_metrics:
        avg_acc = np.mean([m['accuracy'] for m in all_metrics])
        avg_auc = np.mean([m['auc'] for m in all_metrics])
        avg_far = np.mean([m['far'] for m in all_metrics])
        avg_frr = np.mean([m['frr'] for m in all_metrics])

        print("\n===== Aggregated (Vote-" + str(group_size) + ") Across All Targets =====")
        print(f"Avg Accuracy: {avg_acc:.2f}%")
        print(f"Avg AUC:      {avg_auc:.4f}")
        print(f"Avg FAR:      {avg_far:.4f}")
        print(f"Avg FRR:      {avg_frr:.4f}")


if __name__ == "__main__":
    main()
