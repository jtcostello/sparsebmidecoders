import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge
from .training import corrcoef


class ChannelSelectWrapper(nn.Module):
    """
    Wrapper class to add a channel selection layer to a model.
    """
    def __init__(self, core_model, input_size, sigmoid_scalar=500.0, l1_weight=0.2):
        super().__init__()
        self.core_model = core_model
        self.sigmoid_scalar = sigmoid_scalar        # scalar to multiply the gates by to make the sigmoid steeper
        self.l1_weight = l1_weight
        # self.gates_zero = nn.Parameter(torch.zeros(input_size))
        self.gates_zero = nn.Parameter(1e-6 * torch.randn(input_size))  # add a small amount of noise to break symmetry
        self.channel_freeze_mask = nn.Parameter(torch.ones(input_size), requires_grad=False)
        self.input_bn = nn.BatchNorm1d(input_size)

    @property
    def gates(self):
        return nn.Sigmoid()(self.sigmoid_scalar * self.gates_zero) * self.channel_freeze_mask

    def forward(self, x, *args, **kwargs):
        # pass through the gates & BN, and then through the normal model
        # x has shape (batch_size, features, sequence_length)
        x = x * self.gates.view(1, -1, 1)
        x = self.input_bn(x)
        return self.core_model(x, *args, **kwargs)

    # def __getattr__(self, name):
    #     # if attribute (like a function) is not found in the wrapper try to get it from the wrapped model
    #     return getattr(self.core_model, name)

    def compute_loss(self, output, target):
        mse_loss = nn.MSELoss()(output, target)
        l1_loss = torch.sum(torch.abs(self.gates))  # l1 regularization on the gates
        return mse_loss + self.l1_weight * l1_loss

    def freeze_channel_fraction(self, fraction, verbose=True):
        # function freezes the bottom percentage of channels
        with torch.no_grad():
            sorted_gates, _ = torch.sort(self.gates)
            threshold = sorted_gates[int(fraction * len(sorted_gates))]
            self.channel_freeze_mask[self.gates.abs() < threshold] = 0

        # Use register_hook to multiply the gradients with the mask during the backward pass
        self.gates.register_hook(lambda grad: grad * self.channel_freeze_mask)

        if verbose:
            self.print_ignored_inputs(threshold)

    def remove_specific_channels(self, channel_nums):
        with torch.no_grad():
            self.channel_freeze_mask[channel_nums] = 0
        self.gates.register_hook(lambda grad: grad * self.channel_freeze_mask)


    def get_num_dropped_chans(self, threshold=1e-10, verbose=False):
        with torch.no_grad():
            num_ignored = (torch.abs(self.gates) < threshold).sum().item()
            if verbose:
                print(f"Number of ignored inputs: {num_ignored}")
            return num_ignored


class ChannelSelectLayer(nn.Module):
    """
    Layer can be used to learn the important inputs. It applies a weighted gate to each input and batchnorm.
    Very similar to the ChannelSelectWrapper but is a layer instead of a wrapper.
    """

    def __init__(self, input_size, sigmoid_scalar=500.0):
        super().__init__()
        self.gates_zero = nn.Parameter(1e-6 * torch.randn(input_size))  # add a small amount of noise to break symmetry
        self.channel_freeze_mask = nn.Parameter(torch.ones(input_size), requires_grad=False)
        self.input_bn = nn.BatchNorm1d(input_size)
        self.sigmoid_scalar = sigmoid_scalar

    @property
    def gates(self):
        return nn.Sigmoid()(self.sigmoid_scalar * self.gates_zero) * self.channel_freeze_mask

    def forward(self, x, *args, **kwargs):
        # x hopefully has shape (batch_size, features, sequence_length)
        x = x * self.gates.view(1, -1, 1)
        x = self.input_bn(x)
        return x

    def freeze_channel_fraction(self, fraction, verbose=True):
        # freezes the bottom fraction of channels
        with torch.no_grad():
            sorted_gates, _ = torch.sort(self.gates)
            threshold = sorted_gates[int(fraction * len(sorted_gates))]
            self.channel_freeze_mask[self.gates.abs() < threshold] = 0

        # Use register_hook to multiply the gradients with the mask during the backward pass
        self.gates.register_hook(lambda grad: grad * self.channel_freeze_mask)

        if verbose:
            self.print_ignored_inputs(threshold)

    def get_num_dropped_chans(self, threshold=1e-10, verbose=False):
        with torch.no_grad():
            num_ignored = (torch.abs(self.gates) < threshold).sum().item()
            if verbose:
                print(f"Number of ignored inputs: {num_ignored}")
            return num_ignored


class MultidayChannelSelectWrapper(nn.Module):
    """
    Wrapper class to add multiple channel selection layers to a model. Each layer could correspond to a different
    amount of channels dropped or different days.
    """
    def __init__(self, core_model, is_rnn, input_size, core_input_size, num_days, sigmoid_scalar=500.0, l1_weight=0.1):

        super().__init__()
        self.core_model = core_model
        self.is_rnn = is_rnn
        self.num_days = num_days
        self.core_input_size = core_input_size
        self.input_size = input_size
        self.l1_weight = l1_weight

        # setup day-specific input layers - USE JUST A GATE AND A SINGLE LINEAR LAYER FOR NOW
        # note: nn.Linear expects shape (*, Hin) and returns shape (*, Hout)
        self.input_layers = nn.ModuleList(
            nn.Sequential(
                ChannelSelectLayer(input_size, sigmoid_scalar=sigmoid_scalar),
                ReshaperLayer(),
                nn.Linear(input_size, core_input_size),
            ) for _ in range(num_days)
        )

    def forward(self, x, h=None, day_idx=0):
        """NOTE: here we force day_idx to be only one day so that whole batches can be passed through the SparseInput batchnorm layer"""

        # pass through this day's input layers
        assert isinstance(day_idx, int)
        x_afterinput = self.input_layers[day_idx](x)

        # pass through core model
        x_afterinput = x_afterinput.permute(0, 2, 1)  # reshape to (batch_size, num_features, conv_size) to match default model input

        if self.is_rnn:
            return self.core_model(x_afterinput, h)
        else:
            return self.core_model(x_afterinput)

    def freeze_core(self, freeze=True):
        """Freezes the core model so that its weights are not updated during training"""
        for p in self.core_model.parameters():
            p.requires_grad = not freeze

    def compute_loss(self, output, target):
        mse_loss = nn.MSELoss()(output, target)
        l1_loss_sum = 0
        for layer in self.input_layers:
            l1_loss_sum += torch.sum(torch.abs(layer[0].gates))  # l1 regularization on the gates
        return mse_loss + self.l1_weight * l1_loss_sum

    def freeze_channel_fraction(self, fraction_list, verbose=True):
        # function freezes the bottom percentage of channels
        for layer, fraction in zip(self.input_layers, fraction_list):
            layer[0].freeze_channel_fraction(fraction, verbose)

    def print_ignored_inputs(self):
        for layer in self.input_layers:
            layer[0].print_ignored_inputs()

    def get_num_dropped_chans(self, threshold=1e-10, verbose=False):
        with torch.no_grad():
            num_dropped_list = []
            for layer in self.input_layers:
                num_dropped_list.append(layer[0].get_num_dropped_chans(threshold, verbose))
            return num_dropped_list


class ReshaperLayer(nn.Module):
    """Reshapes the neural input to the correct shape for linear layers in the multi-gate model"""
    def forward(self, x, *args, **kwargs):
        # x should have shape (batch_size, input_size, sequence_length)
        # we want to return (batch_size, sequence_length, input_size) since the linear layer expects (*, in_features)
        return x.permute(0, 2, 1)


def calc_linear_chan_order(x, y):
    """Function to train a ridge regression model on each channel to predict the outputs, and then sort the channels
    based on the average correlation with the outputs.
    x: shape (num_samps, num_chans, num_hist_bins)
    y: shape (num_samps, num_outputs)
    """
    avgcorrs = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        # train a ridge regression model on this channel (use the hist_bins as features)
        ridge = Ridge(alpha=1.0)
        ridge.fit(x[:, i, :], y)
        yhat = ridge.predict(x[:, i, :])
        avgcorrs[i] = np.mean(np.abs(corrcoef(y, yhat)))

    sorted_chans = np.argsort(avgcorrs)[::-1]
    sorted_corrs = avgcorrs[sorted_chans]

    return sorted_chans, sorted_corrs
