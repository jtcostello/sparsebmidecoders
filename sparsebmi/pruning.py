import torch.nn.utils.prune as prune


class ModelPruner:
    """
    Helper class to prune weights in a pytorch model.
    We use the torch prune module for pruning, which adds a mask to the weights specifying which ones are pruned. This
    works using a forward hook that multiplies the weights by the mask before passing them to the next layer.
    At the end, call the finalize() method to remove the mask and leave the pruned weights.

    The `finalize_every` argument is used to periodically remove the pruning masks to free up memory. When doing global
    pruning the torch prune module stores a history of the masks for each layer and can max out memory. This isn't an
    issue when doing layer-wise pruning since it doesn't store a history of the masks. More info:
        https://discuss.pytorch.org/t/global-pruning-uses-increasing-amounts-of-memory/195231/2
        These are stored in model.layer._forward_pre_hooks and managed by the PruningContainer class.
    """

    def __init__(self, model, parameters_to_prune, prune_layerwise=False, prev_prune_pct=0, finalize_every=-1):
        """
        args:
            model: pytorch model
            parameters_to_prune: list of tuples of the form (module, name) where module is a module in the model and
                name is the name of the parameter to prune.
                Ex: [(model.conv1, 'weight'), (model.fc1, 'weight'), (model.fc2, 'bias')]
            prune_layerwise: if True, prunes each layer independently. If False, prunes all layers together.
            prev_prune_pct: the amount of pruning that has already been done to the model, in percent
            finalize_every: if > 0, periodically remove the pruning mask to free up memory. If memory is not an issue,
                set to -1 to never remove the mask. If memory is an issue, set this to a small number like 10 or 100.
        """
        self.model = model
        self.prune_layerwise = prune_layerwise
        self.prev_amount_pct = prev_prune_pct
        self.parameters_to_prune = parameters_to_prune
        self.num_iters = 0
        self.finalize_every = finalize_every

    def prune(self, pruning_pct):
        # calc amount to prune
        # the torch pruning module expects the amount to prune to be a fraction of the weights remaining
        # for example, if 75% are already pruned and we want 90%, we need to prune 0.6 of the remaining 25% of weights
        amount = (pruning_pct - self.prev_amount_pct) / (100 - self.prev_amount_pct)
        self.prev_amount_pct = pruning_pct

        self.num_iters += 1
        if self.prune_layerwise:
            for module, name in self.parameters_to_prune:
                prune.l1_unstructured(module, name=name, amount=amount)
        else:
            prune.global_unstructured(
                self.parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
            if (self.finalize_every > 0) and (self.num_iters % self.finalize_every == 0):
                # periodically remove the pruning mask to free up memory
                self.finalize()

    def finalize(self):
        # Permanently removes the pruning mask to leave the pruned weights
        try:
            for module, name in self.parameters_to_prune:
                prune.remove(module, name)
            self.prev_amount_pct = 0    # since with the mask removed torch.prune isn't aware they're pruned
        except:
            pass


def get_percent_pruned(model, verbose=True):
    """
    Prints the percentage of weights that are pruned in each layer of the model, and the total percentage pruned.

    Note: this may return 0% pruned if the model hasn't been finalized yet.
    """
    total_pruned = 0
    total_weights = 0
    for name, param in model.named_parameters():
        # if name in self.params_to_prune:
        weights = param.data.cpu()
        pct_pruned = (weights == 0).sum().item() / weights.numel()
        if verbose:
            print(f"{name}: {pct_pruned*100:.1f}% pruned out of {weights.numel()}")
        total_weights += weights.numel()
        total_pruned += (weights == 0).sum().item()
    if verbose:
        print(f"Total: {total_pruned/total_weights*100:.1f}% pruned out of {total_weights}\n")
    return 100 * total_pruned/total_weights


def count_non_zero_weights(model):
    """
    Returns the number of non-zero weights in the model.
    """
    total_non_zero = 0
    total_weights = 0
    for name, param in model.named_parameters():
        weights = param.data.cpu()
        total_weights += weights.numel()
        total_non_zero += (weights != 0).sum().item()
    return total_non_zero, total_weights
