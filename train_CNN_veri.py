import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sn
import random
from scipy.interpolate import interp1d # New import for interpolation

# Import from your CNN module
from CNN.CNN_network_veri import VerificationCNN, train_verification_model, evaluate_verification_model

# Import for machine configuration
from functions.load_machine_config import load_machine_config


# Constants for expected data dimensions (based on 65x6)
EXPECTED_TIME_STEPS = 65
EXPECTED_FEATURES = 6

def load_feature_data(work_directory, tablename, username, fingername, featurename_folder, num_instances):
    feature_data_list = []
    for idx in range(1, num_instances + 1):
        file_path = os.path.join(
            work_directory, "segments", tablename, username, fingername,
            featurename_folder, f"touchscreen_featureVector_{idx}.csv"
        )
        try:
            df = pd.read_csv(file_path, header=None)
            if df.empty:
                # print(f"Warning: File {file_path} is empty. Skipping.")
                continue

            current_file_rows, num_features_in_file = df.shape

            if num_features_in_file != EXPECTED_FEATURES:
                print(f"Warning: File {file_path} has {num_features_in_file} features, expected {EXPECTED_FEATURES}. Skipping.")
                continue

            # Remove rows with any NaNs
            df_cleaned = df.dropna(axis=0, how='any')
            rows_after_nan_removal = df_cleaned.shape[0]
            rows_removed_count = current_file_rows - rows_after_nan_removal

            if rows_after_nan_removal == 0:
                # print(f"Warning: File {file_path} has 0 rows after NaN removal. Skipping.")
                continue
            
            processed_data_np = None

            # Determine if interpolation/resampling is needed
            # It's needed if NaNs were removed AND/OR the row count is not the EXPECTED_TIME_STEPS
            needs_processing = (rows_removed_count > 0) or (rows_after_nan_removal != EXPECTED_TIME_STEPS)

            if not needs_processing: # Already correct shape and no NaNs changed row count relevantly
                processed_data_np = df_cleaned.values
            elif rows_after_nan_removal >= 2: # Sufficient rows for linear interpolation
                # if rows_removed_count > 0:
                #     print(f"Note: Removed {rows_removed_count} NaN rows from {file_path} (had {current_file_rows} rows).")
                # if rows_after_nan_removal != EXPECTED_TIME_STEPS:
                #     print(f"Note: Resampling/Interpolating {file_path} from {rows_after_nan_removal} rows to {EXPECTED_TIME_STEPS} rows.")

                current_data_for_interp = df_cleaned.values
                interpolated_data_np = np.zeros((EXPECTED_TIME_STEPS, EXPECTED_FEATURES))
                
                # Proportional x-coordinates for current and target data
                x_current_proportional = np.linspace(0, 1, rows_after_nan_removal)
                x_target_proportional = np.linspace(0, 1, EXPECTED_TIME_STEPS)

                for col_idx in range(EXPECTED_FEATURES):
                    y_current_feature = current_data_for_interp[:, col_idx]
                    # 'linear' is robust. 'slinear', 'quadratic', 'cubic' would need more points.
                    interp_func = interp1d(x_current_proportional, y_current_feature, kind='linear', fill_value="extrapolate")
                    interpolated_data_np[:, col_idx] = interp_func(x_target_proportional)
                
                processed_data_np = interpolated_data_np
            else: # rows_after_nan_removal < 2, cannot interpolate
                print(f"Warning: File {file_path} has < 2 rows ({rows_after_nan_removal}) after NaN cleanup for interpolation. Skipping.")
                continue
            
            # Final check on the processed data
            if processed_data_np is not None:
                if not np.isfinite(processed_data_np).all():
                    print(f"Warning: Non-finite values found in processed data for {file_path} post-interpolation. Replacing with zeros.")
                    processed_data_np = np.nan_to_num(processed_data_np, nan=0.0, posinf=0.0, neginf=0.0)
                
                if processed_data_np.shape == (EXPECTED_TIME_STEPS, EXPECTED_FEATURES):
                    feature_data_list.append(processed_data_np)
                else:
                    print(f"Error: Processed data for {file_path} has unexpected shape {processed_data_np.shape} after all steps. Expected ({EXPECTED_TIME_STEPS},{EXPECTED_FEATURES}). Skipping.")
            
        except FileNotFoundError:
            pass # Common, skip silently
        except pd.errors.EmptyDataError:
            # print(f"Warning: File {file_path} is empty or invalid CSV. Skipping.")
            pass
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
            
    return feature_data_list

def normalize_all_data(X, method='zscore'):
    X_normalized = X.copy()
    if method == 'none':
        return X_normalized
    if method == 'zscore':
        g_mean, g_std = np.mean(X_normalized), np.std(X_normalized)
        X_normalized = (X_normalized - g_mean) / (g_std if g_std > 1e-8 else 1.0)
    elif method == 'minmax':
        g_min, g_max = np.min(X_normalized), np.max(X_normalized)
        denominator = g_max - g_min
        X_normalized = (X_normalized - g_min) / (denominator if denominator > 1e-8 else 1.0)
        if denominator <= 1e-8: X_normalized = np.zeros_like(X)
    else:
        raise ValueError(f"Unknown normalization: {method}")
    return X_normalized


def prepare_user_verification_data(target_data_raw_list, all_user_finger_data, target_key):
    num_genuine = len(target_data_raw_list)
    if num_genuine == 0:
        return np.array([]), np.array([])

    final_impostor_pool_raw = []
    target_user_id = target_key.split('_')[0]

    other_users_data_map = {}
    for key, data_list_for_user_finger in all_user_finger_data.items():
        user_id_of_sample_source = key.split('_')[0]
        if user_id_of_sample_source == target_user_id:
            continue
        if user_id_of_sample_source not in other_users_data_map:
            other_users_data_map[user_id_of_sample_source] = []
        other_users_data_map[user_id_of_sample_source].extend(data_list_for_user_finger)

    if not other_users_data_map:
        pass

    guaranteed_impostors_from_step2 = []
    guaranteed_samples_identifiers = set() # Stores (user_id, original_idx)

    for user_id, samples_list in other_users_data_map.items():
        if samples_list:
            chosen_idx = random.randrange(len(samples_list))
            chosen_sample = samples_list[chosen_idx]
            guaranteed_impostors_from_step2.append(chosen_sample)
            guaranteed_samples_identifiers.add((user_id, chosen_idx))

    final_impostor_pool_raw.extend(guaranteed_impostors_from_step2)
    num_needed_more = num_genuine - len(final_impostor_pool_raw)

    if num_needed_more > 0:
        all_other_user_samples_flat_pool_exclusive = []
        for user_id, samples_list in other_users_data_map.items():
            for idx, sample in enumerate(samples_list):
                if (user_id, idx) not in guaranteed_samples_identifiers:
                    all_other_user_samples_flat_pool_exclusive.append(sample)
        
        if not all_other_user_samples_flat_pool_exclusive:
            if final_impostor_pool_raw and num_needed_more > 0:
                for _ in range(num_needed_more):
                    final_impostor_pool_raw.append(random.choice(final_impostor_pool_raw))
        else:
            random.shuffle(all_other_user_samples_flat_pool_exclusive)
            can_pick_without_replacement = min(num_needed_more, len(all_other_user_samples_flat_pool_exclusive))
            final_impostor_pool_raw.extend(all_other_user_samples_flat_pool_exclusive[:can_pick_without_replacement])
            num_still_needed_after_no_replace = num_genuine - len(final_impostor_pool_raw)
            if num_still_needed_after_no_replace > 0:
                for _ in range(num_still_needed_after_no_replace):
                    final_impostor_pool_raw.append(random.choice(all_other_user_samples_flat_pool_exclusive))

    if len(final_impostor_pool_raw) > num_genuine:
        random.shuffle(final_impostor_pool_raw)
        final_impostor_pool_raw = final_impostor_pool_raw[:num_genuine]

    genuine_samples_processed = [np.expand_dims(s, axis=0) for s in target_data_raw_list]
    impostor_samples_processed = [np.expand_dims(s, axis=0) for s in final_impostor_pool_raw]

    X_list, y_list = [], []
    if genuine_samples_processed:
        X_list.extend(genuine_samples_processed)
        y_list.extend(np.ones(len(genuine_samples_processed)))
    if impostor_samples_processed:
        X_list.extend(impostor_samples_processed)
        y_list.extend(np.zeros(len(impostor_samples_processed)))

    if not X_list:
        return np.array([]), np.array([])
    return np.array(X_list), np.array(y_list)


def plot_results(metrics_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(12, 10))
    has_valid_roc_data = False
    for metrics in metrics_list: # Ensure metrics is not None
        if metrics and 'fpr' in metrics and 'tpr' in metrics and 'auc' in metrics and not np.isnan(metrics['auc']):
             plt.plot(metrics['fpr'], metrics['tpr'], lw=2, alpha=0.8, label=f"{metrics.get('target_key','Unknown')} (AUC = {metrics['auc']:.3f})")
             has_valid_roc_data = True
    if has_valid_roc_data:
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.legend(loc="lower right", fontsize=10)
    else:
        plt.text(0.5, 0.5, "No valid ROC data to plot", horizontalalignment='center', verticalalignment='center')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12); plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.savefig(os.path.join(save_dir, 'roc_curves_all_targets.png')); plt.close()

    target_keys = [m.get('target_key','Skipped') for m in metrics_list if m]
    fars_raw = [m.get('far', np.nan) for m in metrics_list if m]
    frrs_raw = [m.get('frr', np.nan) for m in metrics_list if m]
    accuracies_raw = [m.get('accuracy', np.nan) for m in metrics_list if m]

    fars_plot = np.nan_to_num(fars_raw)
    frrs_plot = np.nan_to_num(frrs_raw)
    accuracies_plot_norm = np.nan_to_num(np.array(accuracies_raw)/100.0)

    if not target_keys: # If all metrics were None or empty
        print("Warning: No target keys found in metrics list for plotting FAR/FRR/Accuracy.")
        return pd.DataFrame()


    x_indices = np.arange(len(target_keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(14, len(target_keys)*0.7), 8))
    ax.bar(x_indices - width, fars_plot, width, label='FAR', color='salmon')
    ax.bar(x_indices, frrs_plot, width, label='FRR', color='skyblue')
    ax.bar(x_indices + width, accuracies_plot_norm, width, label='Accuracy/100', color='lightgreen')
    ax.set_xlabel('User-Finger Combinations', fontsize=12); ax.set_ylabel('Rate / Value', fontsize=12)
    ax.set_title('FAR, FRR, and Accuracy by Target', fontsize=14)
    ax.set_xticks(x_indices); ax.set_xticklabels(target_keys, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=10); ax.grid(axis='y', linestyle='--'); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'far_frr_accuracy_comparison.png')); plt.close()

    summary_data = {
        'Target': target_keys,
        'Accuracy (%)': [round(a, 2) if not np.isnan(a) else 'N/A' for a in accuracies_raw],
        'AUC': [round(m.get('auc',np.nan), 3) if m and not np.isnan(m.get('auc',np.nan)) else 'N/A' for m in metrics_list],
        'FAR': [round(f, 3) if not np.isnan(f) else 'N/A' for f in fars_raw],
        'FRR': [round(r, 3) if not np.isnan(r) else 'N/A' for r in frrs_raw]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(save_dir, 'performance_summary.csv'), index=False)
    return summary_df

def main():
    config = load_machine_config()
    work_dir_base = config["data_dir"]
    if not work_dir_base.endswith(os.path.sep): work_dir_base += os.path.sep
    work_directory = os.path.join(work_dir_base, "Touch_Vibration")

    tablenames = ["table1"]
    usernames = ["crystal", "james", "jason", "jinwei", "kevin", "rongwei", "ruxin", "will"]
    fingernames = ["right", "left"]
    featurenames_subfolder = ["touchscreen2"] 
    num_instances_to_load = 20

    tablename = tablenames[0]
    featurename_folder = featurenames_subfolder[0]
    device = config["compdev"]
    print(f"Using device: {device}")

    batch_size = 8 
    epochs = 1000
    normalization_method = 'zscore'
    test_split_ratio = 0.5

    print("Loading all user-finger data (NaN rows removed, interpolated to 65x6)...")
    all_user_finger_data = {}
    total_files_processed_successfully = 0
    for username in usernames:
        for fingername in fingernames:
            target_key = f"{username}_{fingername}"
            # load_feature_data now handles NaN removal and interpolation to EXPECTED_TIME_STEPS x EXPECTED_FEATURES
            feature_list = load_feature_data(
                work_directory, tablename, username, fingername, featurename_folder, num_instances_to_load
            )
            if feature_list:
                # All arrays in feature_list should now be EXPECTED_TIME_STEPS x EXPECTED_FEATURES
                all_user_finger_data[target_key] = feature_list
                total_files_processed_successfully += len(feature_list)
    
    if not all_user_finger_data:
        print(f"CRITICAL ERROR: No data loaded/processed successfully. Check paths and data file integrity. Base path: '{work_directory}'")
        return
    print(f"Data loading complete. Successfully processed {total_files_processed_successfully} files into {len(all_user_finger_data)} user-finger datasets.")

    results_dir_name = f"norm_{normalization_method}_e{epochs}_t{int(test_split_ratio*100)}_b{batch_size}_interpNaN"
    results_dir = os.path.join(work_directory, "verification_results", results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")
    metrics_list = []

    for target_key, target_data_raw_list in all_user_finger_data.items():
        if not target_data_raw_list:
            metrics_list.append({'target_key': target_key, 'accuracy': np.nan, 'auc': np.nan, 'far': np.nan, 'frr': np.nan, 'conf_matrix': np.zeros((2,2), dtype=int), 'fpr': np.array([0,1]), 'tpr': np.array([0,1]), 'precision': np.array([0,1]), 'recall': np.array([0,1])})
            continue

        print(f"\n{'-'*25} Processing Target: {target_key} {'-'*25}")
        X, y = prepare_user_verification_data(target_data_raw_list, all_user_finger_data, target_key)

        if X.ndim == 0 or X.shape[0] == 0:
            print(f"Skipping {target_key}: Insufficient data after preparation (X shape: {X.shape}).")
            metrics_list.append({'target_key': target_key, 'accuracy': np.nan, 'auc': np.nan, 'far': np.nan, 'frr': np.nan, 'conf_matrix': np.zeros((2,2), dtype=int), 'fpr': np.array([0,1]), 'tpr': np.array([0,1]), 'precision': np.array([0,1]), 'recall': np.array([0,1])})
            continue

        unique_labels, counts = np.unique(y, return_counts=True)
        label_dist_str = ", ".join([f"Class {int(label)}: {count}" for label, count in zip(unique_labels, counts)])
        print(f"Data for {target_key}: X_shape={X.shape}, y_shape={y.shape}. Labels: {label_dist_str}")

        if len(unique_labels) < 2:
            print(f"Warning: {target_key} has only one class ({label_dist_str}). Skipping training.")
            cm_single = confusion_matrix(y,y, labels=[0,1]) if len(y)>0 else np.zeros((2,2),dtype=int)
            metrics_list.append({'target_key': target_key, 'accuracy': np.nan, 'auc': np.nan, 'far': np.nan, 'frr': np.nan, 'conf_matrix': cm_single, 'fpr': np.array([0,1]), 'tpr': np.array([0,1]), 'precision': np.array([0,1]), 'recall': np.array([0,1])})
            continue

        X_normalized = normalize_all_data(X, method=normalization_method)
        try:
            x_train, x_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test_split_ratio, random_state=42, stratify=y)
        except ValueError as e:
            print(f"Warning: Stratify failed for {target_key} ({e}). Using non-stratified split.")
            x_train, x_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test_split_ratio, random_state=42)

        if x_train.shape[0] < batch_size or x_test.shape[0] < 1 : # check if enough samples for at least one batch / test
            print(f"Skipping {target_key}: Insufficient samples after split for training/testing (Train: {x_train.shape[0]}, Test: {x_test.shape[0]}).")
            metrics_list.append({'target_key': target_key, 'accuracy': np.nan, 'auc': np.nan, 'far': np.nan, 'frr': np.nan, 'conf_matrix': np.zeros((2,2), dtype=int), 'fpr': np.array([0,1]), 'tpr': np.array([0,1]), 'precision': np.array([0,1]), 'recall': np.array([0,1])})
            continue
        
        x_train_t, x_test_t = torch.from_numpy(x_train).float(), torch.from_numpy(x_test).float()
        y_train_t, y_test_t = torch.from_numpy(y_train).float(), torch.from_numpy(y_test).float()
        
        cnn = VerificationCNN(time_steps=EXPECTED_TIME_STEPS, batch_size=min(batch_size, x_train.shape[0]), epochs=epochs) # Adjust batch_size if train set is smaller
        train_loader, test_loader = cnn.prepare_data_loaders(x_train_t, x_test_t, y_train_t, y_test_t)

        if not cnn.is_initialized and x_train_t.shape[0] > 0:
            with torch.no_grad(): cnn(x_train_t[0:1].to(device))
        elif not cnn.is_initialized:
            print(f"Error: Cannot initialize model for {target_key}. Skipping.")
            metrics_list.append({'target_key': target_key, 'accuracy': np.nan, 'auc': np.nan, 'far': np.nan, 'frr': np.nan, 'conf_matrix': np.zeros((2,2), dtype=int), 'fpr': np.array([0,1]), 'tpr': np.array([0,1]), 'precision': np.array([0,1]), 'recall': np.array([0,1])})
            continue
            
        print(f"Training {target_key} (Epochs: {epochs}, Batch: {cnn.batch_size})...")
        start_time = time.time()
        train_verification_model(cnn, train_loader, device=device, learning_rate=1e-4)
        print(f"Training for {target_key} took {time.time()-start_time:.2f}s.")

        metrics = evaluate_verification_model(cnn, test_loader, target_key, device=device)
        metrics_list.append(metrics)
        torch.save(cnn.state_dict(), os.path.join(results_dir, f"model_{target_key}.pt"))

        target_plot_dir = os.path.join(results_dir, "individual_plots", target_key)
        os.makedirs(target_plot_dir, exist_ok=True)

        if metrics and not np.isnan(metrics.get('auc', np.nan)):
            plt.figure(figsize=(8,6))
            plt.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2, label=f'ROC (AUC={metrics["auc"]:.3f})')
            plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--'); plt.xlim([0,1]); plt.ylim([0,1.05])
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC Curve - {target_key}'); plt.legend(loc="lower right")
            plt.savefig(os.path.join(target_plot_dir, f'roc_{target_key}.png')); plt.close()

        if metrics and 'conf_matrix' in metrics and isinstance(metrics['conf_matrix'], np.ndarray):
            cm = metrics['conf_matrix']
            if cm.shape == (2,2):
                cm_sum_rows = cm.sum(axis=1)[:, np.newaxis]
                cm_norm = np.zeros_like(cm, dtype=float)
                for i in range(cm.shape[0]):
                    if cm_sum_rows[i] > 0: cm_norm[i,:] = cm[i,:].astype('float') / cm_sum_rows[i]
                df_cm = pd.DataFrame(cm_norm, index=['True Impostor (0)','True Genuine (1)'], columns=['Pred Impostor (0)','Pred Genuine (1)'])
                plt.figure(figsize=(8,6)); sn.heatmap(df_cm, annot=True, fmt=".2%", cmap="Blues", cbar=True, annot_kws={"size":12})
                plt.title(f'Confusion Matrix (Normalized) - {target_key}', fontsize=14)
                plt.ylabel('True Label',fontsize=12); plt.xlabel('Predicted Label',fontsize=12); plt.tight_layout()
                plt.savefig(os.path.join(target_plot_dir, f'cm_norm_{target_key}.png')); plt.close()

    if metrics_list:
        print("\n" + "="*30 + " Generating Overall Results " + "="*30)
        valid_metrics_list = [m for m in metrics_list if m and not np.isnan(m.get('accuracy',np.nan))]
        if not valid_metrics_list: print("No valid metrics from any target to summarize."); return

        summary_df = plot_results(valid_metrics_list, results_dir)
        print("\nOverall Performance Summary (per user-finger):\n", summary_df)
        user_avg_metrics = {}
        for uname in usernames:
            user_metrics = [m for m in valid_metrics_list if m['target_key'].startswith(f"{uname}_")]
            if user_metrics:
                acc = [m['accuracy'] for m in user_metrics if not np.isnan(m.get('accuracy', np.nan))]
                auc = [m.get('auc',np.nan) for m in user_metrics if not np.isnan(m.get('auc', np.nan))]
                far = [m.get('far',np.nan) for m in user_metrics if not np.isnan(m.get('far', np.nan))]
                frr = [m.get('frr',np.nan) for m in user_metrics if not np.isnan(m.get('frr', np.nan))]
                user_avg_metrics[uname] = {
                    'Avg_Accuracy': np.mean(acc) if acc else np.nan, 'Avg_AUC': np.mean(auc) if auc else np.nan,
                    'Avg_FAR': np.mean(far) if far else np.nan, 'Avg_FRR': np.mean(frr) if frr else np.nan,
                }
        if user_avg_metrics:
            user_summary_df = pd.DataFrame.from_dict(user_avg_metrics, orient='index').round(3)
            user_summary_df.index.name = 'User'
            user_summary_df.to_csv(os.path.join(results_dir, 'user_level_summary.csv'))
            print("\nUser-Level Average Performance:\n", user_summary_df)
            if 'Avg_Accuracy' in user_summary_df.columns:
                valid_user_acc = user_summary_df['Avg_Accuracy'].dropna()
                if not valid_user_acc.empty:
                    print(f"\n{'*'*70}\nOverall Average Accuracy Across Users: {valid_user_acc.mean():.2f}%\n{'*'*70}")

            user_summary_df_plot = user_summary_df.reset_index().copy()
            for col_metric in ['Avg_AUC', 'Avg_Accuracy', 'Avg_FAR', 'Avg_FRR']:
                if col_metric in user_summary_df_plot.columns: user_summary_df_plot[col_metric] = user_summary_df_plot[col_metric].fillna(0)
            
            plt.figure(figsize=(max(12, len(user_summary_df_plot)*0.8), 7))
            x_users_indices = np.arange(len(user_summary_df_plot))
            bar_width = 0.2
            plt.bar(x_users_indices - bar_width*1.5, user_summary_df_plot['Avg_AUC'], bar_width, label='Avg AUC', color='teal', alpha=0.85)
            plt.bar(x_users_indices - bar_width*0.5, user_summary_df_plot['Avg_Accuracy']/100.0, bar_width, label='Avg Acc/100', color='coral', alpha=0.85)
            plt.bar(x_users_indices + bar_width*0.5, user_summary_df_plot['Avg_FAR'], bar_width, label='Avg FAR', color='lightcoral', alpha=0.85)
            plt.bar(x_users_indices + bar_width*1.5, user_summary_df_plot['Avg_FRR'], bar_width, label='Avg FRR', color='lightblue', alpha=0.85)
            plt.xlabel('Users',fontsize=12); plt.ylabel('Avg Value',fontsize=12); plt.title('Avg Performance by User',fontsize=14)
            plt.xticks(x_users_indices, user_summary_df_plot['User'], rotation=45, ha="right", fontsize=10)
            plt.legend(fontsize=10); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'user_level_metrics_comparison.png')); plt.close()
    else: print("No models were successfully trained/evaluated. No summary results.")
    print(f"\nProcessing finished. All results saved in: {results_dir}")

if __name__ == "__main__":
    create_dummy_data = False # Set True to generate dummy data
    if create_dummy_data:
        print("Setting up dummy 65x6 data for testing...")
        dummy_data_root_path = "./dummy_data_for_cnn/" 
        dummy_cfg_for_path = {"data_dir": dummy_data_root_path} 
        dummy_work_dir_base = dummy_cfg_for_path["data_dir"]
        if not os.path.exists(dummy_work_dir_base): os.makedirs(dummy_work_dir_base)
        if not dummy_work_dir_base.endswith(os.path.sep): dummy_work_dir_base += os.path.sep
        dummy_work_dir = os.path.join(dummy_work_dir_base, "Touch_Vibration")
        if not os.path.exists(dummy_work_dir): os.makedirs(dummy_work_dir)
        dummy_users = ["userA", "userB", "userC", "userD"] 
        dummy_fingers = ["right", "left"]; dummy_table = "table1"; dummy_feature_folder = "touchscreen2"
        for user in dummy_users:
            for finger in dummy_fingers:
                feature_path = os.path.join(dummy_work_dir, "segments", dummy_table, user, finger, dummy_feature_folder)
                os.makedirs(feature_path, exist_ok=True)
                num_dummy_instances = np.random.randint(5, 16)
                for i in range(1, num_dummy_instances + 1):
                    # Create a CSV with some NaNs for testing NaN handling
                    raw_data = np.random.rand(EXPECTED_TIME_STEPS, EXPECTED_FEATURES)
                    if i % 3 == 0 and EXPECTED_TIME_STEPS > 5: # Introduce NaNs in some files
                        nan_rows = np.random.choice(EXPECTED_TIME_STEPS, size=np.random.randint(1,5), replace=False)
                        nan_cols = np.random.choice(EXPECTED_FEATURES, size=np.random.randint(1,3), replace=False)
                        for r_idx in nan_rows:
                            for c_idx in nan_cols:
                                raw_data[r_idx, c_idx] = np.nan
                    pd.DataFrame(raw_data).to_csv(
                        os.path.join(feature_path, f"touchscreen_featureVector_{i}.csv"), header=False, index=False)
        print(f"Dummy 65x6 data (with some NaNs) created under {os.path.abspath(dummy_work_dir)}")
        print(f"IMPORTANT: To use dummy data, 'functions/load_machine_config.py' must return 'data_dir': '{os.path.abspath(dummy_data_root_path)}'")
    main()