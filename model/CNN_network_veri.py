import torch

class VerificationCNN(torch.nn.Module):
    def __init__(self, time_steps, batch_size, epochs):
        super(VerificationCNN, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.time_steps = time_steps
        self.feature_dim = 6
        
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(0, 1))
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(0, 1))
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(0, 1))
        
        self.fc1 = None
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 32)
        self.fc4 = torch.nn.Linear(32, 1)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.sigmoid = torch.nn.Sigmoid()
        self.test_in = None
        self.test_out = None
        self.is_initialized = False

    def _initialize_first_fc_layer(self, x):
        self.flattened_size = x.view(x.size(0), -1).size(1)
        self.fc1 = torch.nn.Linear(self.flattened_size, 512).to(x.device)
        torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(self.fc1.bias, 0)
        self.is_initialized = True

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 1))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 1))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 1))
        x = self.dropout(x)
        x_flat = x.view(x.size(0), -1)
        if not self.is_initialized:
            self._initialize_first_fc_layer(x)
        x = torch.nn.functional.relu(self.fc1(x_flat))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return self.sigmoid(x).squeeze()

    def prepare_data_loaders(self, x_train, x_test, y_train, y_test):
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_in, self.test_out = x_test, y_test
        with torch.no_grad():
            self(x_train[0:1])
        return train_loader, test_loader
