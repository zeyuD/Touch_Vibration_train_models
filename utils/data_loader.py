import os
import numpy as np
import pandas as pd

def load_feature_data(work_directory, sessionname, tablename, username, fingername, featurename, num_instances):
    feature_data = []
    for idx in range(1, num_instances + 1):
        file_path = os.path.join(work_directory, "segments", sessionname, tablename, username, fingername, featurename, f"touchscreen_featureVector_{idx}.csv")
        try:
            data = pd.read_csv(file_path, header=None).values
            data = np.nan_to_num(data, nan=0.0)
            feature_data.append(data)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error: {e}")
    return feature_data

def normalize_all_data(X, method='zscore'):
    X_normalized = X.copy()
    if method == 'zscore':
        mean, std = np.mean(X_normalized), np.std(X_normalized)
        std = 1.0 if std == 0 else std
        return (X_normalized - mean) / std
    elif method == 'minmax':
        min_, max_ = np.min(X_normalized), np.max(X_normalized)
        return (X_normalized - min_) / (max_ - min_) if max_ != min_ else np.zeros_like(X_normalized)
    elif method == 'none':
        return X_normalized
    else:
        raise ValueError("Unknown normalization method")

def prepare_user_verification_data(target_data, all_user_finger_data, target_key):
    genuine = [np.expand_dims(d, 0) for d in target_data]
    genuine_labels = np.ones(len(genuine))
    impostors = []
    for k, v in all_user_finger_data.items():
        if k != target_key:
            impostors.extend(np.expand_dims(d, 0) for d in v)
    impostor_sample = np.random.choice(len(impostors), len(genuine), replace=len(impostors) < len(genuine))
    impostor = [impostors[i] for i in impostor_sample] if isinstance(impostor_sample, np.ndarray) else impostors
    impostor_labels = np.zeros(len(impostor))
    X = np.array(genuine + impostor)
    y = np.concatenate([genuine_labels, impostor_labels])
    return X, y
