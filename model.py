import pennylane as qml
from pennylane import numpy as np
import torch
from lib.models.primitives import *
import matplotlib.pyplot as plt
import tqdm as tqdm
import pandas as pd
import torch.nn as nn


class Regressors():
    def __init__(self, time, alpha, W, q_w, n_qubits, basis, n_bases, spin):
        super(Regressors, self)
        # Initialize
        self.n_qubits = n_qubits
        self.basis = basis
        self.n_bases = n_bases
        self.time = time
        self.alpha = alpha
        self.W = W
        self.spin = spin
        #self.iter = iter

    def eta(self, time, q_w):
        model = Eta(n_qubits = self.n_qubits, n_bases = self.basis)
        y = []
        for t in time:
            t = np.linspace(t,t,1)
            y.append(model.forward(t, q_w, self.alpha))
        y = np.array(y).flatten()
        return y

    def forward(self, t, q_w):
        V = []
        for i in range(self.n_bases):
            v = self.eta(t, q_w[i])
            plt.plot(self.time/(self.spin), v, alpha = 0.2, linewidth = 0.3, color = 'red')
            V.append(v)
        V = np.array(V)
        V = np.matmul(self.W, V)
        return V
