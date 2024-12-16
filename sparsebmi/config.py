
class Config:
    """
    A simple class for storing config parameters.
    Examples:
        Init from a dict:
            cfg = Config({'a': 1, 'b': 2})
        Init from a list of dicts:
            cfg = Config().add_config([default_cfg_a, default_cfg_b])
        Init from kwargs:
            cfg = Config(a=1, b=2)
        Init and update later:
            cfg = Config()
            cfg.a = 1
            cfg.b = 2
        Add a dict to an existing config:
            cfg = Config(a=1)
            cfg.add_config({'b': 2})
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def add_config(self, newdict):
        # newdict can be a dict or a list of dicts
        if isinstance(newdict, dict):
            self.__dict__.update(newdict)
        elif isinstance(newdict, list):
            for d in newdict:
                if not isinstance(d, dict):
                    raise ValueError("newdict must be a dict or list of dicts")
                self.__dict__.update(d)
        else:
            raise ValueError("newdict must be a dict or list of dicts")
        return self

    def __repr__(self):
        return str(self.__dict__)


########################################################################################################################

default_cfg_data = {
    'batch_size': 64,
    'pred_type': 'pv',
    'binsize_ms': 32,
    'train_val_test_split': (0.7, 0.1, 0.2),
    'check_every': 500,
    'normalize_x': True,
    'normalize_y': True,
}

########################################################################################################################
# Some default configs:

default_cfg_rnn = {
    'model_type': 'rnn',
    'hist_bins': 20,
    'hidden_size': 150,
    'num_layers': 1,
    'dropout_p': 0.0,
    'is_rnn': True,

    'num_iters': 1500,
    'iters_before_drop': 50,
    'iters_to_max_drop': 1000,
    'prune_layerwise': False,
    'finalize_every': 100,      # how often to finalize the model to save memory

    'start_lr': 1e-3,
    'end_lr': 1e-6,
    'weight_decay': 2e-4,
}

default_cfg_lstm = {
    'model_type': 'lstm',
    'hist_bins': 20,
    'hidden_size': 350,
    'num_layers': 1,
    'dropout_p': 0.0,
    'is_rnn': True,

    'num_iters': 1500,
    'iters_before_drop': 50,
    'iters_to_max_drop': 1000,
    'prune_layerwise': False,
    'finalize_every': 100,      # how often to finalize the model to save memory

    'start_lr': 1e-3,
    'end_lr': 1e-6,
    'weight_decay': 2e-4,
}


default_cfg_tcn = {
    'model_type': 'tcn',
    'hist_bins': 5,
    'conv_num_filts': 16,
    'layer_size_list': [250, 250, 250, 250],
    'dropout_p': 0.2,
    'is_rnn': False,

    'num_iters': 2000,
    'iters_before_drop': 50,
    'iters_to_max_drop': 1500,
    'prune_layerwise': False,
    'finalize_every': 100,      # how often to finalize the model to save memory

    'start_lr': 0.5e-3,
    'end_lr': 1e-6,
    'weight_decay': 2e-3,
}

########################################################################################################################
# Configs for model pruning:

cfg_rnn_nlb = {
    'model_type': 'rnn',
    'hist_bins': 20,
    'hidden_size': 256,
    'num_layers': 1,
    'dropout_p': 0.0,
    'is_rnn': True,

    'num_iters': 2500,
    'iters_before_drop': 50,
    'iters_to_max_drop': 1250,
    'prune_layerwise': False,
    'finalize_every': 100,      # how often to finalize the model to save memory

    'start_lr': 5e-4,
    'end_lr': 1e-5,
    'weight_decay': 1e-4,
}

cfg_lstm_nlb = {
    'model_type': 'rnn',
    'hist_bins': 20,
    'hidden_size': 256,
    'num_layers': 1,
    'dropout_p': 0.0,
    'is_rnn': True,

    'num_iters': 2500,
    'iters_before_drop': 50,
    'iters_to_max_drop': 1250,
    'prune_layerwise': False,
    'finalize_every': 100,      # how often to finalize the model to save memory

    'start_lr': 1e-3,
    'end_lr': 1e-5,
    'weight_decay': 1e-4,
}

########################################################################################################################
# Configs for channel selection:

cfg_lstm_chan_iterativeprune_nlb = {
    'model_type': 'lstm',
    'hist_bins': 20,
    'hidden_size': 300,
    'num_layers': 1,
    'dropout_p': 0.0,
    'is_rnn': True,
    'l1_weight': 0.1,

    'num_iters': 1500,
    'iters_before_drop': 50,
    'iters_to_max_drop': 750,
    'start_lr': 2e-4,
    'end_lr': 5e-6,
    'weight_decay': 1e-3,
}
