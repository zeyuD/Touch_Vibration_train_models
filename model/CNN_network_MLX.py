import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

class CNNet(nn.Module):
    def __init__(self, time_steps, batch_size, num_class, epochs):
        super().__init__()
        self.batch_size = batch_size
        self.num_class = num_class
        self.epochs = epochs
        self.feature_dim = 6 
        
        # Calculate time dimensions after pooling (kernel=2, stride=2)
        time_after_conv1 = time_steps - 2  # kernel_size=3
        time_after_pool1 = time_after_conv1 // 2  # max_pool size=2
        time_after_conv2 = time_after_pool1 - 2  # kernel_size=3
        time_after_pool2 = time_after_conv2 // 2  # max_pool size=2
        time_after_conv3 = time_after_pool2 - 2  # kernel_size=3 
        time_after_pool3 = time_after_conv3 // 2  # max_pool size=2
        self.final_time_dim = max(1, time_after_pool3)

        # MLX Conv2d default is NHWC. 
        # We treat time_steps as Height and feature_dim as Width.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(0, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(0, 1))
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        self.flattened_size = 128 * self.final_time_dim * self.feature_dim
        
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_class)
        
        self.dropout = nn.Dropout(p=0.2)

    def __call__(self, x):
        # x shape expected: [Batch, Time, Features, Channels] -> [B, 41, 6, 1]
        if len(x.shape) == 3:
            x = mx.expand_dims(x, -1)

        x = nn.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = nn.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = nn.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.reshape(x.shape[0], -1)
        # print("x shape after flattening:", x.shape)  # Debugging line
        
        x = nn.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.relu(self.fc2(x))
        x = self.dropout(x)
        x = nn.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

def loss_fn(model, X, y):
    logits = model(X)
    return mx.mean(nn.losses.cross_entropy(logits, y))


def batch_iterate(batch_size, X, y):
    # Ensure X and y are already mlx arrays for better performance
    X = mx.array(X)
    y = mx.array(y)
    # print(X.shape)
    _, time_steps, features, _ = X.shape 

    perm = mx.array(np.random.permutation(y.size)) # Convert to MLX array
    
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        # Reshape to [Batch, Time, Features, Channels]
        # yield X[ids].reshape(-1, 41, 6, 1), y[ids]
        yield X[ids].reshape(-1, time_steps, features, 1), y[ids]


def train_model(model, x_train, y_train, learning_rate=1e-4):
    model.train()

    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for epoch in range(model.epochs):
        running_loss = 0
        batch_count = 0
        
        for x_batch, y_batch in batch_iterate(model.batch_size, x_train, y_train):
            # MLX handles NaNs similarly, but mx.nan_to_num is available if needed
            loss, grads = loss_and_grad_fn(model, x_batch, y_batch)
            
            # Gradient clipping (why? won't work anyway)
            # grads = optim.clip_grad_norm(grads, 1.0)
            
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            running_loss += loss.item()
            batch_count += 1

            predicted = mx.argmax(model(x_batch), axis=1)
            correct = mx.sum(predicted == y_batch).item()
            total = y_batch.size
            
        print(f"Epoch {epoch+1}: Loss {running_loss/batch_count:.6f}  Accuracy {correct/total*100:.3f}%")


def evaluate_model(model, x_test, y_test, usernames):
    # Set to evaluation mode (deactivates Dropout)
    model.eval() 
    
    # MLX is lazy, so model(x_test) only computes when needed
    logits = model(x_test)
    predictions = mx.argmax(logits, axis=1)
    
    # Convert to numpy for sklearn metrics/confusion matrix
    y_true = np.array(y_test)
    y_pred = np.array(predictions)
    
    print(classification_report(y_true, y_pred, target_names=usernames))