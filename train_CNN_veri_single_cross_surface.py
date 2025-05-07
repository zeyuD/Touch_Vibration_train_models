import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from model.CNN_network_veri import VerificationCNN
from utils.data_loader import load_feature_data, normalize_all_data, prepare_user_verification_data
from utils.train_eval import train_verification_model, evaluate_verification_model
from utils.metrics import plot_results

from sklearn.model_selection import train_test_split

from functions.load_machine_config import load_machine_config


def main():
    config = load_machine_config()
    work_directory = os.path.join(config["data_dir"], "Touch_Vibration")
    device = config["compdev"]
    print(f"Using device: {device}")

    # Experiment settings
    sessionname = "cross_surface"
    tablename = "table1"
    featurename = "touchscreen2"
    usernames = ["crystal", "james", "jason", "jinwei", "kevin", "rongwei", "ruxin", "will"]
    fingernames = ["right", "left"]
    num_instances = 20
    batch_size = 8
    epochs = 500
    normalization_method = 'zscore'

    # Load all user-finger data
    print("Loading feature data...")
    all_data = {}
    for user in usernames:
        for finger in fingernames:
            key = f"{user}_{finger}"
            data = load_feature_data(work_directory, sessionname, tablename, user, finger, featurename, num_instances)
            if data:
                all_data[key] = data
                print(f"{key}: {len(data)} samples")
            else:
                print(f"Warning: {key} has no data")

    # Create result output directory
    result_dir = os.path.join(work_directory, f"verification_results_{normalization_method}")
    os.makedirs(result_dir, exist_ok=True)

    metrics_list = []

    # Train and evaluate per user-finger
    for target_key, target_data in all_data.items():
        print(f"\n{'='*30}\nTraining model for {target_key}\n{'='*30}")
        X, y = prepare_user_verification_data(target_data, all_data, target_key)
        if len(X) == 0:
            print(f"Skipping {target_key} due to no data")
            continue

        X = normalize_all_data(X, method=normalization_method)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        x_train = torch.from_numpy(x_train).float().to(device)
        x_test = torch.from_numpy(x_test).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)
        y_test = torch.from_numpy(y_test).float().to(device)

        cnn = VerificationCNN(time_steps=65, batch_size=batch_size, epochs=epochs).to(device)
        train_loader, test_loader = cnn.prepare_data_loaders(x_train, x_test, y_train, y_test)

        start_time = time.time()
        train_verification_model(cnn, train_loader)
        print(f"Training time: {time.time() - start_time:.2f}s")

        metrics = evaluate_verification_model(cnn, test_loader, target_key)
        metrics_list.append(metrics)

        # Save raw scores and labels for vote-5 aggregation
        raw_save_dir = os.path.join(result_dir, target_key)
        os.makedirs(raw_save_dir, exist_ok=True)
        np.save(os.path.join(raw_save_dir, "raw_scores.npy"), metrics["raw_scores"])
        np.save(os.path.join(raw_save_dir, "raw_labels.npy"), metrics["raw_labels"])


        # Save model
        model_path = os.path.join(result_dir, f"{target_key}_{tablename}_{featurename}_model.pt")
        torch.save(cnn.state_dict(), model_path)

        # Save confusion matrix
        cm_path = os.path.join(result_dir, target_key)
        os.makedirs(cm_path, exist_ok=True)
        plt.figure(figsize=(6, 5))
        df_cm = pd.DataFrame(metrics['conf_matrix'], index=["Impostor", "Genuine"], columns=["Impostor", "Genuine"])
        sn = __import__('seaborn')
        sn.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix for {target_key}")
        plt.savefig(os.path.join(cm_path, "confusion_matrix.png"))
        plt.close()

    # Summary & Aggregation
    if metrics_list:
        summary_df = plot_results(metrics_list, result_dir)
        print(summary_df)

        # Average over users
        user_summary = []
        for user in usernames:
            user_metrics = [m for m in metrics_list if m["target_key"].startswith(user)]
            if user_metrics:
                user_summary.append({
                    "User": user,
                    "Avg_Accuracy": np.mean([m['accuracy'] for m in user_metrics]),
                    "Avg_AUC": np.mean([m['auc'] for m in user_metrics]),
                    "Avg_FAR": np.mean([m['far'] for m in user_metrics]),
                    "Avg_FRR": np.mean([m['frr'] for m in user_metrics])
                })
        user_summary_df = pd.DataFrame(user_summary)
        user_summary_df.to_csv(os.path.join(result_dir, "user_level_summary.csv"), index=False)
        print("\nUser-Level Summary:\n", user_summary_df)

        # Print overall average across all users
        avg_overall = {
            "Avg_Accuracy": user_summary_df["Avg_Accuracy"].mean(),
            "Avg_AUC": user_summary_df["Avg_AUC"].mean(),
            "Avg_FAR": user_summary_df["Avg_FAR"].mean(),
            "Avg_FRR": user_summary_df["Avg_FRR"].mean()
        }
        print("\n=== Average Across All Users ===")
        for key, value in avg_overall.items():
            print(f"{key}: {value:.4f}")


        # Plot averages
        plt.figure(figsize=(10, 6))
        x = np.arange(len(user_summary_df))
        width = 0.2
        plt.bar(x - width*1.5, user_summary_df["Avg_AUC"], width, label="AUC")
        plt.bar(x - width/2, user_summary_df["Avg_Accuracy"] / 100, width, label="Accuracy / 100")
        plt.bar(x + width/2, user_summary_df["Avg_FAR"], width, label="FAR")
        plt.bar(x + width*1.5, user_summary_df["Avg_FRR"], width, label="FRR")
        plt.xticks(x, user_summary_df["User"], rotation=45)
        plt.legend()
        plt.title("Average Performance Metrics by User")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "user_level_comparison.png"))
        plt.close()
    else:
        print("No models were successfully trained.")

if __name__ == "__main__":
    main()
