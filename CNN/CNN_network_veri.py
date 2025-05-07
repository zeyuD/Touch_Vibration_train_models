import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim 
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

class VerificationCNN(nn.Module):
    def __init__(self, time_steps, batch_size, epochs):
        super(VerificationCNN, self).__init__()
        # Store configuration parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.time_steps = time_steps  # 65 in your case
        self.feature_dim = 6  # Fixed feature dimension from your data
        
        # Calculate output sizes after convolutions and pooling
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
        self.flattened_size = 128 * self.final_time_dim * self.feature_dim
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)  # Binary output (genuine or impostor)
        
        # Prevent overfitting
        self.dropout = nn.Dropout(p=0.5)
        
        # Sigmoid activation for binary output
        self.sigmoid = nn.Sigmoid()
        
        # Store test data
        self.test_in = None
        self.test_out = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

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
        
        # Apply sigmoid for binary output
        x = self.sigmoid(x)
        
        return x.squeeze()  # Remove extra dimension for binary output

    def prepare_data_loaders(self, x_train, x_test, y_train, y_test):
        """
        Prepare data loaders for training and testing.
        
        Args:
            x_train (torch.Tensor): Training input data
            x_test (torch.Tensor): Testing input data
            y_train (torch.Tensor): Training target data (binary)
            y_test (torch.Tensor): Testing target data (binary)
            
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
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
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

def evaluate_verification_model(model, test_loader):
    """
    Evaluate the verification CNN model.
    
    Args:
        model (VerificationCNN): The trained CNN model
        test_loader (DataLoader): DataLoader with test data
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
            
            # Store outputs and labels for ROC curve
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
            # Calculate accuracy
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Calculate metrics
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    # Import metrics
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_outputs)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_outputs)
    average_precision = average_precision_score(all_labels, all_outputs)
    
    # Confusion matrix
    y_pred = (all_outputs > 0.5).astype(int)
    conf_matrix = confusion_matrix(all_labels, y_pred)
    
    # Print metrics
    print(f'AUC: {roc_auc:.4f}')
    print(f'Average Precision: {average_precision:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    
    # Calculate False Accept Rate (FAR) and False Reject Rate (FRR)
    tn, fp, fn, tp = conf_matrix.ravel()
    far = fp / (fp + tn)  # False Accept Rate
    frr = fn / (fn + tp)  # False Reject Rate
    
    print(f'False Accept Rate (FAR): {far:.4f}')
    print(f'False Reject Rate (FRR): {frr:.4f}')
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    labels = ['Impostor', 'Genuine']
    df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    sn.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')