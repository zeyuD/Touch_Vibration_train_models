import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

class VerificationCNN(torch.nn.Module):
    def __init__(self, time_steps, batch_size, epochs): # time_steps here is the H_in (e.g., 65)
        super(VerificationCNN, self).__init__()
        # Store configuration parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.time_steps = time_steps # e.g., 65

        # Define model layers
        # Input: (N, 1, time_steps, features_in), e.g. (N, 1, 65, 6)
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(0, 1))
        # H_out = H_in - K_h + 1 => 65 - 3 + 1 = 63
        # W_out = W_in - K_w + 2*P_w + 1 => 6 - 3 + 2*1 + 1 = 6
        # Output: (N, 32, 63, 6)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(0, 1))
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(0, 1))

        # Placeholder for flattened size and first FC layer
        self.flattened_size = None
        self.fc1 = None

        # Subsequent fully connected layers
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 32)
        self.fc4 = torch.nn.Linear(32, 1)

        # Prevent overfitting
        self.dropout = torch.nn.Dropout(p=0.2)

        # Sigmoid activation for binary output
        self.sigmoid = torch.nn.Sigmoid()

        # Flag to indicate if the model's first FC layer is initialized
        self.is_initialized = False

    def _initialize_first_fc_layer(self, x_flat):
        """
        Initialize the first fully connected layer based on the actual tensor dimensions after convolutions.
        """
        self.flattened_size = x_flat.size(1)
        print(f"Dynamically determined flattened size: {self.flattened_size}") # e.g. 128 * H_final * W_final

        self.fc1 = torch.nn.Linear(self.flattened_size, 512).to(x_flat.device)

        torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(self.fc1.bias, 0)

        self.is_initialized = True

    def forward(self, x):
        # x expected as (N, 1, time_steps, features), e.g. (N, 1, 65, 6)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
             x = x.unsqueeze(0).unsqueeze(0)

        # Apply convolutions and pooling
        # Layer 1
        x = torch.nn.functional.relu(self.conv1(x)) # (N, 32, 63, 6)
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 1), stride=(2,1)) # (N, 32, 31, 6) H_out = floor((63-2)/2 + 1) = 31
        x = self.dropout(x)

        # Layer 2
        x = torch.nn.functional.relu(self.conv2(x)) # (N, 64, 29, 6) H_out = 31-3+1 = 29
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 1), stride=(2,1)) # (N, 64, 14, 6) H_out = floor((29-2)/2 + 1) = 14
        x = self.dropout(x)

        # Layer 3
        x = torch.nn.functional.relu(self.conv3(x)) # (N, 128, 12, 6) H_out = 14-3+1 = 12
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 1), stride=(2,1)) # (N, 128, 6, 6) H_out = floor((12-2)/2 + 1) = 6
        x = self.dropout(x)
        # Final feature map per sample: (128, 6, 6)

        # Flatten the tensor
        x_flat = x.view(x.size(0), -1) # (N, 128*6*6) = (N, 4608)

        if not self.is_initialized or self.fc1 is None:
            self._initialize_first_fc_layer(x_flat)
        elif self.flattened_size != x_flat.size(1):
             print(f"Warning: Flattened size mismatch. Expected {self.flattened_size}, got {x_flat.size(1)}. Re-initializing fc1.")
             self._initialize_first_fc_layer(x_flat)

        x = torch.nn.functional.relu(self.fc1(x_flat))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)

        return x.squeeze()

    def prepare_data_loaders(self, x_train, x_test, y_train, y_test):
        print(f"{x_train.shape[0]} train samples")
        print(f"{x_test.shape[0]} test samples")

        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.test_in = x_test # Optional storage
        self.test_out = y_test # Optional storage

        if not self.is_initialized and len(train_dataset) > 0:
            with torch.no_grad():
                sample_input = x_train[0:1]
                self(sample_input.to(next(self.parameters()).device)) # Ensure sample is on same device as model

        return train_loader, test_loader

def train_verification_model(model, train_loader, learning_rate=1e-4, max_grad_norm=1.0, device='cpu'):
    model.to(device)
    model.train()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)

    for epoch in range(model.epochs):
        running_loss = 0
        correct = 0
        total = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            if torch.isnan(x_batch).any():
                print(f"Warning: NaN detected in input batch {batch_idx}, epoch {epoch+1}. Replacing with zeros.")
                x_batch = torch.nan_to_num(x_batch, nan=0.0)

            optimizer.zero_grad()
            outputs = model(x_batch)
            
            y_batch_target = y_batch.float()
            if outputs.shape != y_batch_target.shape: # Adjusting for BCELoss compatibility
                if y_batch_target.ndim == outputs.ndim + 1 and y_batch_target.shape[-1] == 1:
                    y_batch_target = y_batch_target.squeeze(-1)
                elif outputs.ndim == y_batch_target.ndim +1 and outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)


            loss = criterion(outputs, y_batch_target)

            if torch.isnan(loss).any():
                print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch_target).sum().item()
            total += y_batch_target.size(0)
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        accuracy = correct * 100 / total if total > 0 else 0
        print(f'Epoch: {epoch+1}/{model.epochs}  Loss: {avg_loss:.6f}  Accuracy: {accuracy:.3f}%')

def evaluate_verification_model(model, test_loader, target_key, device='cpu'):
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            outputs = model(x_batch)
            
            y_batch_eval = y_batch.float()
            if outputs.shape != y_batch_eval.shape: # Adjust for comparison
                 if y_batch_eval.ndim == outputs.ndim + 1 and y_batch_eval.shape[-1] == 1:
                    y_batch_eval = y_batch_eval.squeeze(-1)
                 elif outputs.ndim == y_batch_eval.ndim +1 and outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)

            predicted = (outputs > 0.5).float()

            all_outputs.extend(outputs.cpu().numpy().flatten()) # Ensure 1D arrays for metrics
            all_labels.extend(y_batch_eval.cpu().numpy().flatten())

            correct += (predicted == y_batch_eval).sum().item()
            total += y_batch_eval.size(0)

    accuracy = 100 * correct / total if total > 0 else 0
    print(f'Target: {target_key} - Test Accuracy: {accuracy:.2f}%')

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)

    if len(np.unique(all_labels)) < 2:
        print(f"Warning: Only one class present in y_true for target {target_key}. ROC AUC and Average Precision cannot be robustly calculated.")
        roc_auc = float('nan')
        average_precision = float('nan')
        fpr, tpr = np.array([0, 1]), np.array([0, 1]) 
        precision, recall = np.array([0,1]), np.array([1,0]) if np.sum(all_labels)>0 else np.array([0,0]) # if only negatives, recall is 0
        conf_matrix = confusion_matrix(all_labels, (all_outputs > 0.5).astype(int), labels=[0,1]) if total > 0 else np.zeros((2,2), dtype=int)
    else:
        fpr, tpr, _ = roc_curve(all_labels, all_outputs)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(all_labels, all_outputs)
        average_precision = average_precision_score(all_labels, all_outputs)
        conf_matrix = confusion_matrix(all_labels, (all_outputs > 0.5).astype(int), labels=[0,1])

    if conf_matrix.size == 4:
        tn, fp, fn, tp = conf_matrix.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else: # Should ideally not happen if labels=[0,1] is used in confusion_matrix
        far = float('nan')
        frr = float('nan')
        # Attempt to reconstruct if possible, though confusion_matrix with labels should prevent this
        if total > 0:
            actual_positives = np.sum(all_labels == 1)
            actual_negatives = np.sum(all_labels == 0)
            predicted_positives_indices = (all_outputs > 0.5)
            
            tp = np.sum((all_labels == 1) & (predicted_positives_indices))
            fp = np.sum((all_labels == 0) & (predicted_positives_indices))
            fn = actual_positives - tp
            tn = actual_negatives - fp

            far = fp / actual_negatives if actual_negatives > 0 else 0.0
            frr = fn / actual_positives if actual_positives > 0 else 0.0


    print(f'AUC: {roc_auc:.4f}')
    print(f'Average Precision: {average_precision:.4f}')
    print(f'False Accept Rate (FAR): {far:.4f}')
    print(f'False Reject Rate (FRR): {frr:.4f}')

    metrics = {
        'target_key': target_key, 'accuracy': accuracy, 'auc': roc_auc,
        'avg_precision': average_precision, 'far': far, 'frr': frr,
        'conf_matrix': conf_matrix, 'fpr': fpr, 'tpr': tpr,
        'precision': precision, 'recall': recall
    }
    return metrics