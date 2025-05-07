import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim 
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


class CNNet(nn.Module):
    def __init__(self, time_steps, batch_size, num_class, epochs):
        super(CNNet, self).__init__()
        self.test_in = []
        self.test_out = []
        self.batch_size = batch_size
        self.num_class = num_class
        self.epochs = epochs
        self.time_steps = time_steps  # 41 in your case
        self.feature_dim = 6  # Fixed feature dimension from your data
        
        # Calculate output sizes after convolutions and pooling
        # For time series data, we typically apply convolutions across time
        # Using a kernel of size 3 reduces dimension by 2
        # Max pooling with size 2 reduces dimension by factor of 2
        
        # First conv: 41 -> 39
        # First pool: 39 -> 19
        # Second conv: 19 -> 17
        # Second pool: 17 -> 8
        # Third conv: 8 -> 6
        # Third pool: 6 -> 3
        
        # Dynamically calculate the final feature size
        time_after_conv1 = time_steps - 2  # kernel_size=3
        time_after_pool1 = time_after_conv1 // 2  # max_pool size=2
        time_after_conv2 = time_after_pool1 - 2  # kernel_size=3
        time_after_pool2 = time_after_conv2 // 2  # max_pool size=2
        time_after_conv3 = time_after_pool2 - 2  # kernel_size=3 
        time_after_pool3 = time_after_conv3 // 2  # max_pool size=2
        
        self.final_time_dim = max(1, time_after_pool3)  # Ensure at least 1
        
        # Define model layers - adapted for time series data
        # Input shape: [batch_size, 1, time_steps, feature_dim]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(0, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(0, 1))
        
        # Calculate the flattened size for the fully connected layer
        # After 3 conv layers and 3 pooling layers
        self.flattened_size = 128 * self.final_time_dim * self.feature_dim
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_class)
        
        # Prevent overfitting
        self.dropout = nn.Dropout(p=0.2)
        
        # Store test data
        self.test_in = None
        self.test_out = None

    def forward(self, x):
        # x shape: [batch_size, 1, time_steps, feature_dim]
        # Ensure input is properly shaped with channel dimension
        if x.dim() == 3:
            # If input is [batch_size, time_steps, feature_dim]
            x = x.unsqueeze(1)
        
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(2, 1))  # Pool over time dimension only
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 1))  # Pool over time dimension only
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=(2, 1))  # Pool over time dimension only
        x = self.dropout(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.flattened_size)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

    def prepare_data_loaders(self, x_train, x_test, y_train, y_test):
        """
        Prepare data loaders for training and testing.
        
        Args:
            x_train (torch.Tensor): Training input data
            x_test (torch.Tensor): Testing input data
            y_train (torch.Tensor): Training target data
            y_test (torch.Tensor): Testing target data
            
        Returns:
            tuple: Training and testing data loaders
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
        
        return train_loader, test_loader

# Training function
def train_model(model, train_loader, learning_rate=1e-4, max_grad_norm=1.0):
    """
    Train the CNN model with gradient clipping and NaN detection.
    
    Args:
        model (CNNet): The CNN model to train
        train_loader (DataLoader): DataLoader with training data
        learning_rate (float, optional): Learning rate for optimizer
        max_grad_norm (float, optional): Maximum norm for gradient clipping
    """
    # Set model to training mode
    model.train()
    
    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
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
            loss = criterion(outputs, y_batch)
            
            # Check if loss is NaN
            if torch.isnan(loss).any():
                print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {batch_idx}")
                # Skip this batch
                continue
                
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"Warning: NaN gradient detected at epoch {epoch+1}, batch {batch_idx}")
                optimizer.zero_grad()  # Clear the bad gradients
                continue
                
            # Update weights
            optimizer.step()
            
            # Calculate statistics
            predicted = torch.max(outputs.data, 1)[1]
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            running_loss += loss.item()
            
        # Print epoch statistics
        if total > 0:  # Avoid division by zero
            print(f'Epoch: {epoch+1}  Loss: {running_loss/len(train_loader):.6f}  Accuracy: {correct*100/total:.3f}%')
        else:
            print(f'Epoch: {epoch+1}  Loss: {running_loss/len(train_loader) if len(train_loader) > 0 else 0:.6f}  Accuracy: 0.000%')

# Evaluation function
def evaluate_model(model, test_loader, usernames):
    """
    Evaluate the CNN model.
    
    Args:
        model (CNNet): The trained CNN model
        test_loader (DataLoader): DataLoader with test data
        usernames (list): List of usernames for display
    """
    # Set model to evaluation mode
    model.eval()
    
    correct = 0
    total = 0
    predict_label = []
    true_label = []
    
    for test_inputs, test_labels in test_loader:
        # Forward pass
        outputs = model(test_inputs)
        
        # Get predictions
        predicted = torch.max(outputs, 1)[1]
        
        # Store predictions and true labels
        predict_label.append(predicted)
        true_label.append(test_labels)
        
        # Calculate accuracy
        correct += (predicted == test_labels).sum().item()
        total += test_labels.size(0)
    
    print(f'Correct: {correct}, Test accuracy: {100*correct/total:.3f}%')
    
    # Get all predictions and true labels
    true_l = model.test_out.to('cpu')
    with torch.no_grad():
        pred_l = torch.max(model(model.test_in), 1)[1].to('cpu')
    
    # Create confusion matrix
    array = confusion_matrix(true_l, pred_l)
    print('Confusion Matrix:')
    print(array)
    
    # Get unique labels
    unique_labels = sorted(torch.unique(true_l).tolist())
    
    # Generate target names for classification report
    target_names = [usernames[i] for i in unique_labels]
    
    # Display classification report
    print('Classification Report:')
    print(classification_report(true_l, pred_l, target_names=target_names))
    
    # Normalize confusion matrix for better visualization
    array_norm = np.around(array.astype('float') / np.sum(array, axis=1)[:, None], decimals=2)
    
    # Create dataframe for seaborn heatmap
    df_cm_norm = pd.DataFrame(
        array_norm, 
        index=[usernames[i] for i in unique_labels],
        columns=[usernames[i] for i in unique_labels]
    )
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sn.heatmap(df_cm_norm, annot=True, cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Predicted User')
    plt.xlabel('True User')