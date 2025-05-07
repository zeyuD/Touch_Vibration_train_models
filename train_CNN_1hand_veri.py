import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import seaborn as sn

def load_feature_data(work_directory, tablename, username, fingername, featurename, num_instances):
    """
    Load time-series feature data from CSV files with NaN handling.
    
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
        method (str): Normalization method ('zscore', 'minmax', or 'none')
        
    Returns:
        numpy.ndarray: Normalized data
    """
    # Make a copy to avoid modifying the original data
    X_normalized = X.copy()
    
    if method == 'none':
        # No normalization, return data as is
        print("No normalization applied")
        return X_normalized
    
    elif method == 'zscore':
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

def prepare_user_verification_data(target_user_data, other_users_data):
    """
    Prepare verification data for a single user.
    
    Args:
        target_user_data (list): Feature data for the target user
        other_users_data (list): Feature data from all other users
        
    Returns:
        tuple: X_data, y_labels where y_labels are binary (1=genuine, 0=impostor)
    """
    if not target_user_data:
        print("Warning: No data for target user")
        return np.array([]), np.array([])
    
    # Prepare genuine samples from target user
    genuine_samples = []
    for sample in target_user_data:
        # Add channel dimension (1 channel)
        # Original: [41, 90] -> Reshaped: [1, 41, 90]
        genuine_samples.append(np.expand_dims(sample, axis=0))
    
    # Create label array for genuine samples
    genuine_labels = np.ones(len(genuine_samples))
    
    # Prepare impostor samples from other users
    # We'll try to get a balanced dataset with equal numbers of genuine and impostor samples
    num_genuine = len(genuine_samples)
    
    # Flatten all other users' data into a single list
    all_impostor_samples = []
    for user_samples in other_users_data:
        for sample in user_samples:
            all_impostor_samples.append(np.expand_dims(sample, axis=0))
    
    # Randomly select impostor samples to match the number of genuine samples
    if len(all_impostor_samples) > num_genuine:
        # If we have more impostor samples than needed, randomly select some
        impostor_indices = np.random.choice(
            len(all_impostor_samples), 
            num_genuine, 
            replace=False
        )
        impostor_samples = [all_impostor_samples[i] for i in impostor_indices]
    else:
        # If we don't have enough impostor samples, use them all and randomly repeat some
        impostor_indices = np.random.choice(
            len(all_impostor_samples), 
            num_genuine - len(all_impostor_samples), 
            replace=True
        )
        impostor_samples = all_impostor_samples.copy()
        impostor_samples.extend([all_impostor_samples[i] for i in impostor_indices])
    
    # Create label array for impostor samples
    impostor_labels = np.zeros(len(impostor_samples))
    
    # Combine genuine and impostor samples and labels
    X = np.array(genuine_samples + impostor_samples)
    y = np.concatenate([genuine_labels, impostor_labels])
    
    return X, y

class VerificationCNN(torch.nn.Module):
    def __init__(self, time_steps, batch_size, epochs):
        super(VerificationCNN, self).__init__()
        # Store configuration parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.time_steps = time_steps  # 41 in your case
        self.feature_dim = 6  # Fixed feature dimension from your data
        
        # Define model layers
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(0, 1))
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(0, 1))
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(0, 1))
        
        # Placeholder for flattened size
        self.flattened_size = None
        
        # Fully connected layers with placeholder for first layer
        self.fc1 = None
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 32)
        self.fc4 = torch.nn.Linear(32, 1)
        
        # Prevent overfitting
        self.dropout = torch.nn.Dropout(p=0.2)
        
        # Sigmoid activation for binary output
        self.sigmoid = torch.nn.Sigmoid()
        
        # Store test data
        self.test_in = None
        self.test_out = None
        
        # Flag to indicate if the model is initialized
        self.is_initialized = False

    def _initialize_first_fc_layer(self, x):
        """
        Initialize the first fully connected layer based on the actual tensor dimensions.
        """
        # Get the actual flattened size from the tensor
        self.flattened_size = x.view(x.size(0), -1).size(1)
        print(f"Actual flattened size: {self.flattened_size}")
        
        # Create the first fully connected layer with the correct size
        self.fc1 = torch.nn.Linear(self.flattened_size, 512).to(x.device)
        
        # Initialize weights
        torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(self.fc1.bias, 0)
        
        self.is_initialized = True

    def forward(self, x):
        # Ensure input is properly shaped with channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Apply convolutions
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 1))
        x = self.dropout(x)
        
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 1))
        x = self.dropout(x)
        
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 1))
        x = self.dropout(x)
        
        # Flatten the tensor
        x_flat = x.view(x.size(0), -1)
        
        # Initialize the first FC layer if not done yet
        if not self.is_initialized:
            self._initialize_first_fc_layer(x)
        
        # Apply fully connected layers
        x = torch.nn.functional.relu(self.fc1(x_flat))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        
        # Apply sigmoid for binary output
        x = self.sigmoid(x)
        
        return x.squeeze()  # Remove extra dimension for binary output

    def prepare_data_loaders(self, x_train, x_test, y_train, y_test):
        """
        Prepare data loaders for training and testing.
        """
        print(f"{x_train.shape[0]} train samples")
        print(f"{x_test.shape[0]} test samples")

        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Store test data for evaluation
        self.test_in = x_test
        self.test_out = y_test
        
        # Do a single forward pass to initialize the network dimensions
        if len(train_dataset) > 0:
            with torch.no_grad():
                self(x_train[0:1])
        
        return train_loader, test_loader

def train_verification_model(model, train_loader, learning_rate=1e-4, max_grad_norm=1.0):
    """
    Train the verification CNN model.
    
    Args:
        model (VerificationCNN): The CNN model to train
        train_loader (DataLoader): DataLoader with training data
        learning_rate (float): Learning rate for the optimizer
        max_grad_norm (float): Maximum gradient norm for clipping
    """
    # Set model to training mode
    model.train()
    
    # Define loss and optimizer for binary classification
    criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    
    # Training loop
    for epoch in range(model.epochs):
        running_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # Check for NaN in input
            if torch.isnan(x_batch).any():
                print(f"Warning: NaN detected in input batch {batch_idx}, replacing with zeros")
                x_batch = torch.nan_to_num(x_batch, nan=0.0)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x_batch)
            
            # Calculate loss
            loss = criterion(outputs, y_batch.float())
            
            # Check if loss is NaN
            if torch.isnan(loss).any():
                print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {batch_idx}")
                continue
                
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Update weights
            optimizer.step()
            
            # Calculate statistics
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            running_loss += loss.item()
            
        # Print epoch statistics
        if total > 0:
            print(f'Epoch: {epoch+1}  Loss: {running_loss/len(train_loader):.6f}  Accuracy: {correct*100/total:.3f}%')
        else:
            print(f'Epoch: {epoch+1}  Loss: {running_loss/len(train_loader) if len(train_loader) > 0 else 0:.6f}  Accuracy: 0.000%')

def evaluate_verification_model(model, test_loader, username):
    """
    Evaluate the verification CNN model and generate performance metrics.
    
    Args:
        model (VerificationCNN): The trained CNN model
        test_loader (DataLoader): DataLoader with test data
        username (str): Name of the user for this model
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            # Forward pass
            outputs = model(x_batch)
            
            # Calculate predictions
            predicted = (outputs > 0.5).float()
            
            # Store outputs and labels for metrics
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
            # Calculate accuracy
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f'User: {username} - Test Accuracy: {accuracy:.2f}%')
    
    # Calculate metrics
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_outputs)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_outputs)
    average_precision = average_precision_score(all_labels, all_outputs)
    
    # Confusion matrix
    y_pred = (all_outputs > 0.5).astype(int)
    conf_matrix = confusion_matrix(all_labels, y_pred)
    
    # Calculate False Accept Rate (FAR) and False Reject Rate (FRR)
    tn, fp, fn, tp = conf_matrix.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Accept Rate
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Reject Rate
    
    print(f'AUC: {roc_auc:.4f}')
    print(f'Average Precision: {average_precision:.4f}')
    print(f'False Accept Rate (FAR): {far:.4f}')
    print(f'False Reject Rate (FRR): {frr:.4f}')
    
    # Return metrics as a dictionary
    metrics = {
        'username': username,
        'accuracy': accuracy,
        'auc': roc_auc,
        'avg_precision': average_precision,
        'far': far,
        'frr': frr,
        'conf_matrix': conf_matrix,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }
    
    return metrics

def plot_results(metrics_list, save_dir):
    """
    Plot comparison results for all users.
    
    Args:
        metrics_list (list): List of metrics dictionaries for each user
        save_dir (str): Directory to save plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot ROC curves for all users
    plt.figure(figsize=(12, 10))
    for metrics in metrics_list:
        plt.plot(
            metrics['fpr'], 
            metrics['tpr'], 
            lw=2, 
            label=f"{metrics['username']} (AUC = {metrics['auc']:.2f})"
        )
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves for All Users')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    
    # Plot FAR and FRR for all users
    usernames = [m['username'] for m in metrics_list]
    fars = [m['far'] for m in metrics_list]
    frrs = [m['frr'] for m in metrics_list]
    accuracies = [m['accuracy'] for m in metrics_list]
    
    x = np.arange(len(usernames))
    width = 0.25
    
    plt.figure(figsize=(14, 8))
    plt.bar(x - width, fars, width, label='FAR')
    plt.bar(x, frrs, width, label='FRR')
    plt.bar(x + width, [a/100 for a in accuracies], width, label='Accuracy/100')
    
    plt.xlabel('Users')
    plt.ylabel('Rate')
    plt.title('FAR, FRR, and Accuracy by User')
    plt.xticks(x, usernames, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'far_frr_comparison.png'))
    
    # Create a summary table
    summary_data = {
        'User': usernames,
        'Accuracy (%)': accuracies,
        'AUC': [m['auc'] for m in metrics_list],
        'FAR': fars,
        'FRR': frrs
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(save_dir, 'performance_summary.csv'), index=False)
    
    print(f"Results saved to {save_dir}")
    return summary_df

def main():
    # Import configuration
    from functions.load_machine_config import load_machine_config
    config = load_machine_config()
    work_directory = config["data_dir"] + "Touch_Vibration/"
    
    # Configure dataset parameters
    tablenames = ["table3"]
    usernames = ["crystal", "james", "jason", "jinwei", "kevin", "rongwei", "ruxin", "will"]
    fingernames = ["left"]
    featurenames = ["touchscreen2"]
    num_instances = 20
    
    # For simplicity, using only first tablename and fingername
    tablename = tablenames[0]
    fingername = fingernames[0]
    featurename = featurenames[0]
    
    # Configure device (CPU or GPU)
    device = config["compdev"]
    print(f"Using device: {device}")
    
    # Configure training parameters
    batch_size = 8
    epochs = 500
    normalization_method = 'zscore'  # Options: 'zscore', 'minmax', 'none'
    
    # Load data for all users
    print("Loading data for all users...")
    all_user_data = {}
    for username in usernames:
        feature_data = load_feature_data(
            work_directory, tablename, username, fingername, featurename, num_instances
        )
        if feature_data:
            all_user_data[username] = feature_data
            print(f"Loaded {len(feature_data)} samples for user {username}")
        else:
            print(f"Warning: No data loaded for user {username}")
    
    # Create results directory
    results_dir = os.path.join(work_directory, f"verification_results_{normalization_method}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Train and evaluate a model for each user
    metrics_list = []
    
    for username in usernames:
        if username not in all_user_data or not all_user_data[username]:
            print(f"Skipping user {username} due to insufficient data")
            continue
            
        print(f"\n{'='*50}")
        print(f"Training model for user: {username}")
        print(f"{'='*50}")
        
        # Prepare verification data for this user
        target_user_data = all_user_data[username]
        other_users_data = [data for name, data in all_user_data.items() if name != username]
        
        X, y = prepare_user_verification_data(target_user_data, other_users_data)
        
        if len(X) == 0:
            print(f"Skipping user {username} due to insufficient data after preparation")
            continue
            
        print(f"Prepared verification data with shape X: {X.shape}, y: {y.shape}")
        
        # Apply global normalization
        X = normalize_all_data(X, method=normalization_method)
        
        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        
        # Convert to PyTorch tensors
        x_train = torch.from_numpy(x_train).float().to(device)
        x_test = torch.from_numpy(x_test).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)
        y_test = torch.from_numpy(y_test).float().to(device)
        
        # Initialize verification CNN model
        input_size = 65  # time steps dimension
        cnn = VerificationCNN(input_size, batch_size=batch_size, epochs=epochs)
        cnn = cnn.to(device)
        
        # Prepare data loaders
        train_loader, test_loader = cnn.prepare_data_loaders(x_train, x_test, y_train, y_test)
        
        # Train verification model
        print("Starting training...")
        start_time = time.time()
        train_verification_model(cnn, train_loader)
        train_time = time.time()
        print(f"Training time: {train_time-start_time:.2f} seconds")
        
        # Evaluate verification model
        print("Evaluating model...")
        metrics = evaluate_verification_model(cnn, test_loader, username)
        metrics_list.append(metrics)
        
        # Save model
        model_save_path = os.path.join(
            results_dir, 
            f"{username}_{tablename}_{fingername}_{featurename}_model.pt"
        )
        torch.save(cnn, model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Plot individual user results
        user_results_dir = os.path.join(results_dir, username)
        os.makedirs(user_results_dir, exist_ok=True)
        
        # ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2, 
                 label=f'ROC curve (area = {metrics["auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for User: {username}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(user_results_dir, 'roc_curve.png'))
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        labels = ['Impostor', 'Genuine']
        conf_matrix = metrics['conf_matrix']
        df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
        sn.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f'Confusion Matrix for User: {username}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(user_results_dir, 'confusion_matrix.png'))
    
    # Plot comparative results for all users
    if metrics_list:
        summary_df = plot_results(metrics_list, results_dir)
        print("\nPerformance Summary:")
        print(summary_df)
    else:
        print("No models trained due to insufficient data")

if __name__ == "__main__":
    main()