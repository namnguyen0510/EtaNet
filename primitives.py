import time
import os
import copy

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms

# Pennylane
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt


def H_layer(w):
    for idx in range(w):
        qml.Hadamard(wires=idx)

def RX_layer(w):
    for idx, element in enumerate(w):
        qml.RX(element, wires=idx)

def RY_layer(w):
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def RZ_layer(w):
    for idx, element in enumerate(w):
        qml.RZ(element, wires=idx)

def entangling_layer(nqubits):
    for i in range(0, nqubits-1):
        qml.CNOT(wires=[i, i+1])


def entangling_layer(nqubits):
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

class Basis():
    def __init__(self, n_qubits, q_depth, q_delta = 0.01):
        super(Basis,self)
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        q_delta = q_delta
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, t, weights):
        dev = qml.device("default.qubit", wires=self.n_qubits)
        @qml.qnode(dev, interface="torch")
        def Unitary(t):
            q_weights = weights.reshape(self.q_depth, self.n_qubits)
            H_layer(self.n_qubits)
            for k in range(self.q_depth):
                RX_layer(q_weights[k])
                RY_layer(t)
                entangling_layer(self.n_qubits)
            exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(self.n_qubits)]
            return tuple(exp_vals)
        output = Unitary(t)
        return output



class Eta():
    def __init__(self, n_qubits = 1, n_bases = 4):
        super(Eta, self)
        self.bases = [Basis(n_qubits = n_qubits, q_depth = i) for i in range(1, n_bases+1)]
        self.softmax = nn.Softmax(dim = 0)

    def forward(self, t, q_w, alpha):
        output = []
        for i, b in enumerate(self.bases):
            h = b.forward(t, q_w[i])
            output.append(h)
        output = torch.cat(output, dim = 0).unsqueeze(1)
        alpha = np.array(alpha)
        alpha = torch.from_numpy(alpha)
        alpha = self.softmax(alpha)
        output = torch.matmul(alpha, output)
        return output

    def grad(self, q_w):
        grads = []
        for i, b in enumerate(self.bases):
            h = qml.grad(b)
            grads.append(h)
        return grads
