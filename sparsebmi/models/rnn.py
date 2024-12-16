import torch
import torch.nn as nn


class RNN(nn.Module):
    """
    A general recurrent model that can use VanillaRNN/GRU/LSTM, with a linear layer to the output
    """

    def __init__(self, num_inputs, hidden_size, num_outputs, num_layers, rnn_type='lstm',
                 drop_prob=0, dropout_input=0, add_input_linear=False, device='cpu'):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        rnn_type = rnn_type.lower()
        self.rnn_type = rnn_type
        self.device = device

        if dropout_input:
            self.dropout_input = nn.Dropout(dropout_input)
        else:
            self.dropout_input = None

        if rnn_type == 'rnn':
            self.rnn = nn.RNN(num_inputs, hidden_size, num_layers, dropout=drop_prob, batch_first=True, nonlinearity='relu')
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(num_inputs, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(num_inputs, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
        else:
            raise ValueError(f"rnn_type must be 'rnn', 'gru', or 'lstm', but got {rnn_type}")
        self.fc = nn.Linear(hidden_size, num_outputs)

        self.add_input_linear = add_input_linear
        if add_input_linear:
            self.input_linear = nn.Linear(num_inputs, num_inputs)

    def forward(self, x, h=None,  return_all_tsteps=False):
        """
        x:                  Neural data tensor of shape (batch_size, num_inputs, sequence_length)
        h:                  Hidden state tensor
        return_all_steps:   If true, returns predictions from all timesteps in the sequence. If false, only returns the
                            last step in the sequence.
        """
        x = x.permute(0, 2, 1)  # put in format (batches, sequence length (history), features)

        if self.dropout_input and self.training:
            x = self.dropout_input(x)

        if self.add_input_linear:
            x = self.input_linear(x)

        if h is None:
            h = self.init_hidden(x.shape[0])

        out, h = self.rnn(x, h)
        # out shape:    (batch_size, seq_len, hidden_size) like (64, 20, 350)
        # h shape:      (n_layers, batch_size, hidden_size) like (2, 64, 350)

        if return_all_tsteps:
            out = self.fc(out)  # out now has shape (batch_size, seq_len, num_outs) like (64, 20, 2)
        else:
            out = self.fc(out[:, -1])  # out now has shape (batch_size, num_outs) like (64, 2)
        return out, h

    def init_hidden(self, batch_size):
        if self.rnn_type == 'lstm':
            # lstm - create a tuple of two hidden states
            hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=self.device),
                      torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=self.device))
        else:
            # not an lstm - just a single hidden state vector
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=self.device)
        return hidden

    def get_pruning_params(self):
        # prune all non-bias params
        parameters_to_prune = []
        for i in range(self.num_layers):
            parameters_to_prune.append((self.rnn, f'weight_ih_l{i}'))
            parameters_to_prune.append((self.rnn, f'weight_hh_l{i}'))
        parameters_to_prune.append((self.fc, 'weight'))
        return parameters_to_prune
