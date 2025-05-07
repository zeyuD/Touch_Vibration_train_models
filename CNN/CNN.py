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
    def __init__(self, img_size, batch_size, num_class, epochs):
    # def __init__(self, X_ro, X_hu, Y_ro, Y_hu, batch_size, num_class, epochs):
        super(CNNet, self).__init__()
        # self.X = X
        # self.Y = Y
        self.test_in = []
        self.test_out = []
        self.batch_size = batch_size
        self.num_class = num_class
        self.epochs = epochs
        self.img_size = img_size
        self.view_size = 35
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3)
        self.conv3 = nn.Conv2d(64,128, kernel_size=3)
        self.fc4 = nn.Linear(128*self.view_size*self.view_size, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, num_class)

    def forward(self, x):
        # print(x.shape)
        x0 = x.view(-1,3,self.img_size,self.img_size).float() # batch, views
        # print(x0.shape)
        x0 = F.relu(self.conv1(x0))
        x0 = F.relu(F.max_pool2d(self.conv2(x0), 2))
        x0 = F.dropout(x0, p=0.5, training=self.training)
        x0 = F.relu(F.max_pool2d(self.conv3(x0), 2))
        x0 = F.dropout(x0, p=0.5, training=self.training)
        # print("x0 shape",x0.shape)
        x0 = x0.view(-1,128*self.view_size*self.view_size)
        x0 = F.relu(self.fc4(x0))
        x0 = F.relu(self.fc5(x0))
        x0 = F.relu(self.fc6(x0))
        x0 = self.fc7(x0)

        return x0

    def train(self, x_train, x_test, y_train, y_test):
        # x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.5, random_state=42)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # Pytorch train and test sets
        train = torch.utils.data.TensorDataset(x_train, y_train)
        test = torch.utils.data.TensorDataset(x_test, y_test)
        # data loader
        train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle = True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle = True)
        self.test_in = x_test
        self.test_out = y_test
        return train_loader, test_loader

def fit(model, train_loader):
    # it = iter(train_loader)
    # X_batch, Y_batch = next(it)
    # print(model.forward(X_batch).shape)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(),lr=1e-4)
    for epoch in range(model.epochs):
        running_loss = 0
        correct = 0
        total = 0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # input = torch.autograd.Variable(x_batch).float()
            # target = torch.autograd.Variable(y_batch)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(outputs.data, 1)[1] 
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            running_loss = running_loss + loss.item()
            #print(correct)
        else:
            print('Epoch: {}  Loss: {:.6f}  Accuracy: {:.3f}%'.format(epoch+1, running_loss/len(train_loader), correct*100/total))

def eva(model, test_loader, users):
    correct = 0 
    total = 0
    predict_label = []
    true_label = []
    for test_imgs, test_labels in test_loader:
        #print(test_imgs.shape)
        # test_imgs = torch.autograd.Variable(test_imgs).float()
        outputs = model(test_imgs)
        predicted = torch.max(outputs,1)[1]
        predict_label.append(predicted)
        true_label.append(test_labels)
        correct += (predicted == test_labels).sum().item()
        total += test_labels.size(0)
    print('Correct:', correct, "Test accuracy:{:.3f}% ".format(100*correct/total))

    true_l = model.test_out.to('cpu')
    pred_l = torch.max(model(model.test_in),1)[1].to('cpu')
    array = confusion_matrix(true_l, pred_l)
    print(true_l)
    print(pred_l)
    
   # MPS not yet supported for sklearn
    print('Confusion Matrix')
    print(array)
    print('Classification Report')
    
    woduplicates = []
    for i in true_l.tolist():
        if i not in woduplicates:
            woduplicates.append(i)
    print(woduplicates)

    target_names = []
    for i in woduplicates:
        target_names.append('U'+str(i+1))
    print(classification_report(true_l, pred_l, target_names=target_names))

    # df_cm = pd.DataFrame(array, index = [i for i in "SWZ"],
    #                 columns = [i for i in "SWZ"])
    # plt.figure(figsize = (3,2.4))
    # sn.heatmap(df_cm, annot=True)
    # plt.tight_layout()
    # plt.title('Confusion Matrix')
    # plt.ylabel('Predicted label')
    # plt.xlabel('True label')
    array_norm = np.around(array.astype('float') / np.sum(array,axis=1)[:,None], decimals=2)
    df_cm_norm = pd.DataFrame(array_norm, index = [i for i in target_names],
                    columns = [i for i in target_names])
    plt.figure(figsize = (3,2.4))
    sn.heatmap(df_cm_norm, annot=True)
    # plt.tight_layout()
    plt.title('Confusion Matrix')
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    # plt.show()
