"""Plaintext and MPC models with parameterized activation functions.
Pass activation_fn for hidden layers, pass output_fn to 
apply an activation after the final layer"""

import torch.nn as nn
import crypten.nn as cnn
from functools import partial

class PlainTextCNN(nn.Module):
    def __init__(self, num_classes=10, activation_fn=nn.Sigmoid):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.activation1 = activation_fn()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.activation2 = activation_fn()
        self.pool2 = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.activation3 = activation_fn()
        self.fc2 = nn.Linear(512, num_classes)

        layers = [
            self.conv1,
            self.activation1,
            self.pool1,
            self.conv2,
            self.activation2,
            self.pool2,
            self.flatten,
            self.fc1,
            self.activation3,
            self.fc2
        ]
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class PlainTextMLP(nn.Module):

    def __init__(self, num_classes=10, activation_fn=nn.Sigmoid):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3072, 512)
        self.activation1 = activation_fn()
        self.fc2 = nn.Linear(512, 256)
        self.activation2 = activation_fn()
        self.fc3 = nn.Linear(256, num_classes)

        layers = [
            self.flatten,
            self.fc1,
            self.activation1,
            self.fc2,
            self.activation2,
            self.fc3
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class PlainTextLeNet(nn.Module):
    def __init__(self, num_classes=10, activation_fn=nn.Sigmoid):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.activation1 = activation_fn()
        self.pool1 = nn.AvgPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.activation2 = activation_fn()
        self.pool2 = nn.AvgPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.activation3 = activation_fn()
        
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.activation4 = activation_fn()
        
        self.fc3 = nn.Linear(84, num_classes)

        layers = [
            self.conv1,
            self.bn1,
            self.activation1,
            self.pool1,
            self.conv2,
            self.bn2,
            self.activation2,
            self.pool2,
            self.flatten,
            self.fc1,
            self.bn3,
            self.activation3,
            self.fc2,
            self.bn4,
            self.activation4,
            self.fc3
        ]
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# TODO: Investigate how these activation functions are being approximated exactly, and replace them where required

class MpcFlatten(cnn.Module):
    def forward(self, x):
        return x.flatten(start_dim=1)

class MpcTanh(cnn.Module):
    # Wrapper for Tanh activation in CrypTen.
    def forward(self, x):
        return x.tanh()

class MpcCNN(cnn.Module):

    def __init__(self, num_classes=10, activation_fn=cnn.Sigmoid):
        super().__init__()
        self.conv1 = cnn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.activation1 = activation_fn()
        self.pool1 = cnn.AvgPool2d(2, 2)
        self.conv2 = cnn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.activation2 = activation_fn()
        self.pool2 = cnn.AvgPool2d(2, 2)
        
        self.flatten = MpcFlatten()
        
        self.fc1 = cnn.Linear(64 * 8 * 8, 512)
        self.activation3 = activation_fn()
        self.fc2 = cnn.Linear(512, num_classes)

        layers = [
            self.conv1,
            self.activation1,
            self.pool1,
            self.conv2,
            self.activation2,
            self.pool2,
            self.flatten,
            self.fc1,
            self.activation3,
            self.fc2
        ]

        self.network = cnn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MpcMLP(cnn.Module):
    def __init__(self, num_classes=10, activation_fn=cnn.Sigmoid):
        super().__init__()
        self.flatten = MpcFlatten()
        self.fc1 = cnn.Linear(3072, 512)
        self.activation1 = activation_fn()
        self.fc2 = cnn.Linear(512, 256)
        self.activation2 = activation_fn()
        self.fc3 = cnn.Linear(256, num_classes)

        layers = [
            self.flatten,
            self.fc1,
            self.activation1,
            self.fc2,
            self.activation2,
            self.fc3
        ]

        self.network = cnn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MpcLeNet(cnn.Module):
    def __init__(self, num_classes=10, activation_fn=cnn.Sigmoid):
        super().__init__()
        self.conv1 = cnn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.bn1 = cnn.BatchNorm2d(6)
        self.activation1 = activation_fn()
        self.pool1 = cnn.AvgPool2d(2, 2) 
        
        self.conv2 = cnn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = cnn.BatchNorm2d(16)
        self.activation2 = activation_fn()
        self.pool2 = cnn.AvgPool2d(2, 2)
        
        self.flatten = MpcFlatten()
        
        self.fc1 = cnn.Linear(16 * 6 * 6, 120)
        self.bn3 = cnn.BatchNorm1d(120)
        self.activation3 = activation_fn()
        
        self.fc2 = cnn.Linear(120, 84)
        self.bn4 = cnn.BatchNorm1d(84)
        self.activation4 = activation_fn()
        
        self.fc3 = cnn.Linear(84, num_classes)

        layers = [
            self.conv1,
            self.bn1,
            self.activation1,
            self.pool1,
            self.conv2,
            self.bn2,
            self.activation2,
            self.pool2,
            self.flatten,
            self.fc1,
            self.bn3,
            self.activation3,
            self.fc2,
            self.bn4,
            self.activation4,
            self.fc3
        ]
        
        self.network = cnn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

MPC_MODELS = {
    'MpcCNN_Sigmoid': partial(MpcCNN, activation_fn=cnn.Sigmoid),
    'MpcCNN_Tanh': partial(MpcCNN, activation_fn=MpcTanh),
    'MpcCNN_ReLU': partial(MpcCNN, activation_fn=cnn.ReLU),
    'MpcMLP_Sigmoid': partial(MpcMLP, activation_fn=cnn.Sigmoid),
    'MpcMLP_Tanh': partial(MpcMLP, activation_fn=MpcTanh),
    'MpcMLP_ReLU': partial(MpcMLP, activation_fn=cnn.ReLU),
    'MpcLeNet_Sigmoid': partial(MpcLeNet, activation_fn=cnn.Sigmoid),
    'MpcLeNet_Tanh': partial(MpcLeNet, activation_fn=MpcTanh),
    'MpcLeNet_ReLU': partial(MpcLeNet, activation_fn=cnn.ReLU),
}

PLAINTEXT_MODELS = {
    'PlainTextCNN_Sigmoid': partial(PlainTextCNN, activation_fn=nn.Sigmoid),
    'PlainTextCNN_Tanh': partial(PlainTextCNN, activation_fn=nn.Tanh),
    'PlainTextCNN_ReLU': partial(PlainTextCNN, activation_fn=nn.ReLU),
    'PlainTextMLP_Sigmoid': partial(PlainTextMLP, activation_fn=nn.Sigmoid),
    'PlainTextMLP_Tanh': partial(PlainTextMLP, activation_fn=nn.Tanh),
    'PlainTextMLP_ReLU': partial(PlainTextMLP, activation_fn=nn.ReLU),
    'PlainTextLeNet_Sigmoid': partial(PlainTextLeNet, activation_fn=nn.Sigmoid),
    'PlainTextLeNet_Tanh': partial(PlainTextLeNet, activation_fn=nn.Tanh),
    'PlainTextLeNet_ReLU': partial(PlainTextLeNet, activation_fn=nn.ReLU),
}
