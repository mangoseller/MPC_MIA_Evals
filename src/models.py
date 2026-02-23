"""Plaintext and MPC models with parameterized activation functions.
Pass activation_fn for hidden layers, pass output_fn to 
apply an activation after the final layer

Note on MPC-compatible activations:
- Sigmoid: Approximated via exponential + reciprocal (Newton-Raphson)
- Tanh: Approximated via Chebyshev polynomials or 2*Sigmoid(2x) - 1
- GELU: Approximated via 0.5x(1 + Tanh(sqrt(2/pi)(x + 0.044715x^3)))
"""

import math
import torch.nn as nn
import crypten.nn as cnn
from functools import partial

class PlainTextCNN(nn.Module):
    def __init__(self, num_classes=10, activation_fn=nn.Sigmoid):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.activation1 = activation_fn()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.activation2 = activation_fn()
        self.pool2 = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.activation3 = activation_fn()
        self.fc2 = nn.Linear(1024, num_classes)

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
        self.fc1 = nn.Linear(3072, 1024)
        self.activation1 = activation_fn()
        self.fc2 = nn.Linear(1024, 512)
        self.activation2 = activation_fn()
        self.fc3 = nn.Linear(512, num_classes)

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

class MpcFlatten(cnn.Module):
    def forward(self, x):
        return x.flatten(start_dim=1)

class MpcTanh(cnn.Module):
    # Wrapper for Tanh activation in CrypTen.
    def forward(self, x):
        return x.tanh()

class MpcGELU(cnn.Module):
    """
    GELU activation for CrypTen using the tanh approximation:
    GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    
    CRITICAL: sqrt(2/pi) must be INSIDE the tanh argument. Placing it
    outside removes the dampening that keeps the argument in the range
    where CrypTen's Chebyshev polynomial approximation of tanh is accurate.
    Without dampening, large inputs cause the polynomial to diverge,
    producing catastrophic numerical blowup (loss in the millions).
    
    Approach follows MPCDIFF (NDSS 2024, Appendix A).
    """

    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        self.coeff = 0.044715
    
    def forward(self, x):
        # Clamp inputs to prevent x^3 from pushing the tanh argument outside
        # CrypTen's stable Chebyshev approximation range (~[-8, 8]).
        # For |x| > 5, true GELU ≈ x (positive) or ≈ 0 (negative), so
        # clamping the *inner computation* input doesn't change the function
        # meaningfully but prevents catastrophic polynomial divergence.
        x_clamped = (x / 5.0).hardtanh() * 5.0
        # sqrt(2/pi) MUST multiply the inner expression BEFORE tanh
        inner = self.sqrt_2_over_pi * (x_clamped + self.coeff * x_clamped * x_clamped * x_clamped)
        return 0.5 * x * (1.0 + inner.tanh())

class MpcCNN(cnn.Module):

    def __init__(self, num_classes=10, activation_fn=cnn.Sigmoid):
        super().__init__()
        self.conv1 = cnn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.activation1 = activation_fn()
        self.pool1 = cnn.AvgPool2d(2, 2)
        self.conv2 = cnn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.activation2 = activation_fn()
        self.pool2 = cnn.AvgPool2d(2, 2)

        self.flatten = MpcFlatten()

        self.fc1 = cnn.Linear(128 * 8 * 8, 1024)
        self.activation3 = activation_fn()
        self.fc2 = cnn.Linear(1024, num_classes)

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
        self.fc1 = cnn.Linear(3072, 1024)
        self.activation1 = activation_fn()
        self.fc2 = cnn.Linear(1024, 512)
        self.activation2 = activation_fn()
        self.fc3 = cnn.Linear(512, num_classes)

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
    'MpcCNN_GELU': partial(MpcCNN, activation_fn=MpcGELU),
    'MpcMLP_Sigmoid': partial(MpcMLP, activation_fn=cnn.Sigmoid),
    'MpcMLP_Tanh': partial(MpcMLP, activation_fn=MpcTanh),
    'MpcMLP_GELU': partial(MpcMLP, activation_fn=MpcGELU),
    'MpcLeNet_Sigmoid': partial(MpcLeNet, activation_fn=cnn.Sigmoid),
    'MpcLeNet_Tanh': partial(MpcLeNet, activation_fn=MpcTanh),
    'MpcLeNet_GELU': partial(MpcLeNet, activation_fn=MpcGELU),
}

PLAINTEXT_MODELS = {
    'PlainTextCNN_Sigmoid': partial(PlainTextCNN, activation_fn=nn.Sigmoid),
    'PlainTextCNN_Tanh': partial(PlainTextCNN, activation_fn=nn.Tanh),
    'PlainTextCNN_GELU': partial(PlainTextCNN, activation_fn=nn.GELU),
    'PlainTextMLP_Sigmoid': partial(PlainTextMLP, activation_fn=nn.Sigmoid),
    'PlainTextMLP_Tanh': partial(PlainTextMLP, activation_fn=nn.Tanh),
    'PlainTextMLP_GELU': partial(PlainTextMLP, activation_fn=nn.GELU),
    'PlainTextLeNet_Sigmoid': partial(PlainTextLeNet, activation_fn=nn.Sigmoid),
    'PlainTextLeNet_Tanh': partial(PlainTextLeNet, activation_fn=nn.Tanh),
    'PlainTextLeNet_GELU': partial(PlainTextLeNet, activation_fn=nn.GELU),
}

# Maps each plaintext model name to its MPC counterpart constructor
PLAINTEXT_TO_MPC = {
    'PlainTextCNN_Sigmoid':  MPC_MODELS['MpcCNN_Sigmoid'],
    'PlainTextCNN_Tanh':     MPC_MODELS['MpcCNN_Tanh'],
    'PlainTextCNN_GELU':     MPC_MODELS['MpcCNN_GELU'],
    'PlainTextMLP_Sigmoid':  MPC_MODELS['MpcMLP_Sigmoid'],
    'PlainTextMLP_Tanh':     MPC_MODELS['MpcMLP_Tanh'],
    'PlainTextMLP_GELU':     MPC_MODELS['MpcMLP_GELU'],
    'PlainTextLeNet_Sigmoid': MPC_MODELS['MpcLeNet_Sigmoid'],
    'PlainTextLeNet_Tanh':    MPC_MODELS['MpcLeNet_Tanh'],
    'PlainTextLeNet_GELU':    MPC_MODELS['MpcLeNet_GELU'],
}

def mpc_name_from_plaintext(plaintext_name: str) -> str:
    """Convert 'PlainTextCNN_Sigmoid' -> 'MpcCNN_Sigmoid'."""
    return plaintext_name.replace('PlainText', 'Mpc')
