import torch
import time


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


class LinearScheduler:
    """simple linear scheduler for pruning, learning rate, etc. With an optional warmup period"""
    def __init__(self, start, stop, warmup_iters, num_iters_startstop):
        self.start = start                              # value at start
        self.stop = stop                                # value at end
        self.warmup_iters = warmup_iters                # number of iterations to warmup before pruning
        self.num_iters_startstop = num_iters_startstop  # number of iterations to go from start to stop (after warmup)
        self.current_iteration = 0
        self.current_val = self.start

    def step(self):
        self.current_iteration += 1
        if self.current_iteration <= self.warmup_iters:
            self.current_val = self.start
        elif self.current_iteration <= self.warmup_iters + self.num_iters_startstop:
            progress = (self.current_iteration - self.warmup_iters) / self.num_iters_startstop
            self.current_val = self.start + progress * (self.stop - self.start)
        else:
            self.current_val = self.stop

    def get_val(self):
        return self.current_val


class LinearLRScheduler:
    """Linearly decreases the learning rate between start and end. step() should be called every iter."""
    def __init__(self, optimizer, start_lr, end_lr, num_steps):
        self.optimizer = optimizer
        self.scheduler = LinearScheduler(start_lr, end_lr, 0, num_steps)

    def step(self):
        lr = self.scheduler.get_val()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class MultiLinearScheduler:
    """Wrapper class for multiple pruning schedulers. Returns a list of pruning factors"""
    def __init__(self, scehduler_list):
        self.scheduler_list = scehduler_list

    def step(self):
        for scheduler in self.scheduler_list:
            scheduler.step()

    def get_val(self):
        return [scheduler.get_val() for scheduler in self.scheduler_list]


def print_model_param_names(model):
    print("")
    for name, param in model.named_parameters():
        print(name)
    print("")


def print_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
