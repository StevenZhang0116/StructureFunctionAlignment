import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
import math
import scipy.io
import random 
import time
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns

class CTRNN(nn.Module):
    """Continuous-time RNN.

    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, Tau=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = Tau
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        self.input2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        pre_activation = self.input2h(input) + self.h2h(hidden) 
        h_new = torch.relu(hidden * self.oneminusalpha +
                           pre_activation * self.alpha)
        return h_new.to(input.device)

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        output = torch.stack(output, dim=0)
        return output, hidden


class CTRNN_Net(nn.Module):
    """Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        rnn_activity, _ = self.rnn(x)
        out = self.fc(rnn_activity)
        return out, rnn_activity


for i in range(50):

    input_size = 5     
    hidden_size = 60   
    output_size = 3    
    sequence_length = 30000

    rnn = CTRNN_Net(input_size, hidden_size, output_size)

    batch_size = 1  
    random_input = torch.randn(batch_size, sequence_length, input_size)

    output, hidden_states = rnn(random_input)

    hidden_states_np = hidden_states.detach().numpy().squeeze() 
    rnn_weight = rnn.rnn.h2h.weight.detach().numpy()

    eigenvalues = np.linalg.eigvals(rnn_weight)    
    radius = np.max(np.abs(eigenvalues))

    correlation_activity = np.corrcoef(hidden_states_np.T)  

    scipy.io.savemat(f"zz_data_rnn/rnn_connectome_out_{i}.mat", {'connectome': rnn_weight})
    scipy.io.savemat(f"zz_data_rnn/rnn_connectome_in_{i}.mat", {'connectome': rnn_weight.T})
    scipy.io.savemat(f"zz_data_rnn/rnn_activity_{i}.mat", {'activity': hidden_states_np.T})
