import torch.nn as nn
import torch.nn.functional as F


class TCN(nn.Module):
    """
    Temporal convolutional network with multiple hidden layers, similar to that of Willsey et al. 2022.
    Convolutional layer over time, then multiple feedforward layers, then linear output layer.

    This model is elsewhere called `NNConvNet_MultiSize`.
    """

    def __init__(self, num_inputs, hist_bins, conv_num_filts, layer_size_list, num_outputs, dropout_p):
        super().__init__()

        self.layer_size_list = layer_size_list

        # convolutional input layer
        self.bncn = nn.BatchNorm1d(num_inputs)
        self.cn = nn.Conv1d(hist_bins, conv_num_filts, 1, bias=True)

        # middle feedforward layer(s)
        middle_size_list = [num_inputs * conv_num_filts] + layer_size_list
        self.hiddenlayers = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(prevsize),
                                                         nn.Linear(prevsize, nextsize),
                                                         nn.Dropout(p=dropout_p))
                                           for prevsize, nextsize in zip(middle_size_list[:-1], middle_size_list[1:])])

        # linear output layer
        self.bnout = nn.BatchNorm1d(layer_size_list[-1])
        self.fcout = nn.Linear(layer_size_list[-1], num_outputs)

    def forward(self, x):
        # x should have shape (batch_size, num_inputs, hist_bins)

        # conv layer
        x = self.bncn(x)
        x = self.cn(x.permute(0, 2, 1))
        x = x.flatten(start_dim=1, end_dim=-1)

        # middle layers
        for layer in self.hiddenlayers:
            x = F.relu(layer(x))    # BN -> linear -> DO -> relu

        # output layer
        output = self.fcout(self.bnout(x))
        return output               # shape (batch_size, num_outputs)

    def get_pruning_params(self):
        # prune linear and conv weights (pruning bias and batchnorm leads to errors or poor performance)
        parameters_to_prune = [(self.cn, 'weight'), (self.fcout, 'weight')]
        for hiddenlayer in self.hiddenlayers:
            parameters_to_prune.append((hiddenlayer[1], 'weight'))
        return parameters_to_prune
