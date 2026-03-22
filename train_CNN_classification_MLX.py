import mlx.core as mx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split

# Import your converted MLX model classes
from model import CNN_network_MLX 
from functions.load_machine_config import load_machine_config
from variables import usernames

config = load_machine_config()

def load_feature_data(work_directory, tablename, username, fingername, featurename, num_instances):
    """
    Load time-series feature data from CSV files with NaN handling.
    No feature-wise normalization is applied at this stage.
    
    Args:
        work_directory (str): Base directory path
        tablename (str): Name of the table
        username (str): Name of the user
        fingername (str): Name of the finger used
        featurename (str): Name of the feature
        num_instances (int): Number of instances to load
        
    Returns:
        list: List of loaded feature matrices
    """
    feature_data = []
    
    for idx in range(1, num_instances+1):
        file_path = os.path.join(
            work_directory,
            "segments",
            "cross_surface",
            tablename,
            username,
            fingername,
            featurename,
            f"touchscreen_featureVector_{idx}.csv"
        )
        
        try:
            # Load CSV data (41 time steps × 90 features)
            data = pd.read_csv(file_path, header=None).values
            
            # Replace NaN values with zeros
            data = np.nan_to_num(data, nan=0.0)
            
            # Check for valid data
            if np.isfinite(data).all():
                feature_data.append(data)
            else:
                print(f"Warning: Non-finite values found in {file_path}")
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                feature_data.append(data)
                
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
            continue
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            continue
            
    return feature_data

def normalize_all_data(X, method='zscore'):
    """
    Apply global normalization to all data points collectively.
    
    Args:
        X (numpy.ndarray): Input data with shape [samples, 1, time_steps, features]
        method (str): Normalization method ('zscore' or 'minmax')
        
    Returns:
        numpy.ndarray: Normalized data
    """
    # Make a copy to avoid modifying the original data
    X_normalized = X.copy()
    
    if method == 'zscore':
        # Z-score normalization across all data points
        global_mean = np.mean(X_normalized)
        global_std = np.std(X_normalized)
        
        # Avoid division by zero
        if global_std == 0:
            global_std = 1.0
        
        # Apply normalization globally
        X_normalized = (X_normalized - global_mean) / global_std
        
    elif method == 'minmax':
        # Min-max normalization to [0, 1] range across all data points
        global_min = np.min(X_normalized)
        global_max = np.max(X_normalized)
        
        # Avoid division by zero
        if global_max == global_min:
            X_normalized = np.zeros_like(X_normalized)  # Set all to zero if no range
        else:
            # Apply normalization globally
            X_normalized = (X_normalized - global_min) / (global_max - global_min)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X_normalized

def prepare_user_samples(feature_data, user_idx):
    """
    Prepare input samples for a specific user.
    
    Args:
        feature_data (list): List of feature matrices for a user
        user_idx (int): Index of the user
        
    Returns:
        tuple: Prepared data and labels
    """
    if not feature_data:
        print("Warning: No feature data provided")
        return np.array([]), np.array([])
    
    # Initialize with first two samples
    if len(feature_data) < 2:
        print("Warning: Not enough samples for user")
        return np.array([]), np.array([])
    
    # Convert to proper format for CNN input
    # CNN expects input of shape [batch_size, channels, height, width]
    # Our data is [time_steps, features], so we need to reshape
    samples = []
    labels = []
    
    for i in range(len(feature_data)):
        # Add a channel dimension (1 channel)
        # Original: [41, 90] -> Reshaped: [1, 41, 90]
        # MLX works best with the channel dimension at the end.
        # Change the axis from 0 to -1.
        sample = np.expand_dims(feature_data[i], axis=-1)
        samples.append(sample)
        labels.append(user_idx)
    
    return np.array(samples), np.array(labels)

def combine_all_user_data(all_user_data, user_indices):
    """
    Combine data from all users into training and testing datasets.
    
    Args:
        all_user_data (list): List of [feature_data, username] pairs
        user_indices (dict): Dictionary mapping usernames to indices
        
    Returns:
        tuple: Combined X and Y data
    """
    if not all_user_data:
        print("Error: No user data provided")
        return np.array([]), np.array([])
    
    all_samples = []
    all_labels = []
    
    for feature_data, username in all_user_data:
        user_idx = user_indices[username]
        samples, labels = prepare_user_samples(feature_data, user_idx)
        
        if samples.size > 0:
            all_samples.extend(samples)
            all_labels.extend(labels)
    
    return np.array(all_samples), np.array(all_labels)

work_directory = config["data_dir"] + "Touch_Vibration/"

# Configure dataset parameters
tablenames = ["table1"]
fingernames = ["right"]
featurenames = ["touchscreen2"]
num_instances = 20

# Create a mapping of usernames to indices
user_indices = {username: idx for idx, username in enumerate(usernames)}

# Load data for all users
print("Loading data...")
all_user_data = []

# For simplicity, using only first tablename and fingername
tablename = tablenames[0]
fingername = fingernames[0]
featurename = featurenames[0]

for username in usernames:
    feature_data = load_feature_data(
        work_directory, tablename, username, fingername, featurename, num_instances
    )
    all_user_data.append([feature_data, username])

# 1. Load and Process Data
X, Y = combine_all_user_data(all_user_data, user_indices)
X = normalize_all_data(X, method='zscore')

# 2. Split Data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# 3. Convert to MLX Arrays (MLX manages memory automatically on Apple Silicon)
x_train = mx.array(x_train)
y_train = mx.array(y_train)
x_test = mx.array(x_test)
y_test = mx.array(y_test)

# 4. Initialize MLX Model
# Note: Ensure 'time_steps' matches your data (41 in your CSV logic)
time_steps = x_train.shape[1] 
model = CNN_network_MLX.CNNet(time_steps=time_steps, batch_size=8, num_class=len(usernames), epochs=2000)

# Initialize weights
mx.eval(model.parameters())

# 5. Train Model
print("Starting training on MLX...")
start_time = time.time()

# Using the train_model function we defined in the previous MLX step
CNN_network_MLX.train_model(model, x_train, y_train, learning_rate=1e-4)

train_time = time.time()
print(f"Training time used: {train_time-start_time:.2f} seconds")

# 6. Evaluate
print("Evaluating model...")
CNN_network_MLX.evaluate_model(model, x_test, y_test, usernames)
# # Simple evaluation logic for MLX
# logits = model(x_test)
# predictions = mx.argmax(logits, axis=1)
# accuracy = mx.mean(predictions == y_test)
test_time = time.time()
print(f"Testing time used: {test_time-train_time:.2f} seconds")
# print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

# # 7. Save Model
# # MLX uses save_weights (safetensors/npz) instead of pickling the whole object
# model.save_weights("touch_cnn_model.safetensors")
# print("Model weights saved to touch_cnn_model.safetensors")


plt.show()