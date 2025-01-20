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

import activity_helper

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
        # nn.init.xavier_uniform_(self.h2h.weight)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        pre_activation = self.input2h(input) + self.h2h(hidden) 
        h_new = torch.relu(hidden * self.oneminusalpha + pre_activation * self.alpha)
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

    input_size = 2     
    hidden_size = 100   
    output_size = 2    
    sequence_length = 30000

    rnn = CTRNN_Net(input_size, hidden_size, output_size, Tau=1, dt=1)

    batch_size = 1  
    # random_input = torch.randn(batch_size, sequence_length, input_size)

    scale = 1
    random_input = scale * torch.randn(batch_size, sequence_length, input_size)


    output, hidden_states = rnn(random_input)

    hidden_states_np = hidden_states.detach().numpy().squeeze() 
    rnn_weight = rnn.rnn.h2h.weight.detach().numpy()

    eigenvalues = np.linalg.eigvals(rnn_weight)    
    radius = np.max(np.abs(eigenvalues))

    correlation_activity = np.corrcoef(hidden_states_np.T)  

    scipy.io.savemat(f"zz_data_rnn/rnn_connectome_out_{i}.mat", {'connectome': rnn_weight})
    scipy.io.savemat(f"zz_data_rnn/rnn_connectome_in_{i}.mat", {'connectome': rnn_weight.T})
    scipy.io.savemat(f"zz_data_rnn/rnn_activity_{i}.mat", {'activity': hidden_states_np.T})

    weight_corr_in = np.corrcoef(rnn_weight, rowvar=True)
    weight_corr_out = np.corrcoef(rnn_weight.T, rowvar=True)
    activity_corr = np.corrcoef(hidden_states_np.T, rowvar=True)

    cov_matrix = np.cov(hidden_states_np, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]
    participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    print(f"Participation ratio: {participation_ratio}")

    dim_loader, angle_loader1, _, _ = activity_helper.angles_between_flats_wrap(weight_corr_in, activity_corr, angle_consideration=15)
    dim_loader, angle_loader2, _, _ = activity_helper.angles_between_flats_wrap(weight_corr_out, activity_corr, angle_consideration=15)

    fig, axs = plt.subplots(1,3, figsize=(4*3,4))
    cnt = 0
    for data in [weight_corr_in, weight_corr_out, activity_corr]:
        axs[cnt].hist(data.flatten(), bins=100, alpha=0.5)
        cnt+=1
    plt.savefig(f"zz_data_rnn/hist_{i}.png")

    weight_corr_in_flatten = weight_corr_in.flatten()
    weight_corr_out_flatten = weight_corr_out.flatten()
    activity_corr_flatten = activity_corr.flatten()

    fig, axs = plt.subplots(1,2,figsize=(4*2,4))
    axs[0].scatter(weight_corr_in_flatten, activity_corr_flatten, alpha=0.1)
    axs[1].scatter(weight_corr_out_flatten, activity_corr_flatten, alpha=0.1)
    fig.savefig(f"zz_data_rnn/scatter_{i}.png")
