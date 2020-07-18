import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNNModel(BaseModel):
    def __init__(self, num_classes=9):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 12, kernel_size=16)
        self.conv2 = nn.Conv1d(12, 12, kernel_size=16)
        self.conv3 = nn.Conv1d(12, 12, kernel_size=16)
        # self.batch_norm1 = nn.BatchNorm1d(3000)
        # self.batch_norm2 = nn.BatchNorm1d(3000)
        # self.batch_norm3 = nn.BatchNorm1d(3000)
        self.conv2_drop = nn.Dropout()
        self.fc1 = nn.Linear(12*2236, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = F.dropout(x, 0.5, training=self.training)

        x = self.conv2(x)
        # x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = F.dropout(x, 0.5, training=self.training)

        x = self.conv3(x)
        # x = self.batch_norm3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = F.dropout(x, 0.5, training=self.training)

        # print(x.size())
        x = x.view(-1, 12*2236)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x

class MLP(BaseModel):
    def __init__(self, input_dim, num_classes, n_hid, activation='ReLU'):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.n_hid = n_hid

        fc = []
        fc.append(nn.Linear(self.input_dim, self.n_hid[0]))
        for i in range(1, len(self.n_hid)):
            fc.append(nn.Linear(self.n_hid[i-1], self.n_hid[i]))
        fc.append(nn.Linear(self.n_hid[-1], self.num_classes))

        self.fc = nn.ModuleList(fc)

        # Non linearity
        if activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        elif activation == 'Sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        for i in range(len(self.fc) - 1):
            x = self.act(self.fc[i](x))
        y = self.fc[-1](x)
        return  y