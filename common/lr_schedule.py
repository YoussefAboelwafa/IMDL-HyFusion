"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""
from abc import ABCMeta, abstractmethod


class BaseLR():
    __metaclass__ = ABCMeta

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.curr_lr = 0

    def step(self, cur_iter):
        self.update_lr(cur_iter)
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = self.curr_lr

    @abstractmethod
    def update_lr(self, cur_iter): pass


class PolyLR(BaseLR):
    def __init__(self, optimizer, start_lr, lr_power, total_iters):
        super().__init__(optimizer)
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

    def update_lr(self, cur_iter):
        self.curr_lr = self.start_lr * (
                (1 - float(cur_iter) / self.total_iters) ** self.lr_power)


class WarmUpPolyLR(BaseLR):
    def __init__(self, optimizer, start_lr, lr_power, total_iters, warmup_steps, min_lr=0.0):
        super().__init__(optimizer)
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = float(total_iters)
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def update_lr(self, cur_iter):
        if cur_iter < self.warmup_steps:
            # Linear warmup from 0 to start_lr
            warmup_lr = self.start_lr * (cur_iter / self.warmup_steps)
            self.curr_lr = max(self.min_lr, warmup_lr)
        else:
            # Poly decay from start_lr to min_lr
            poly_decay = ((1 - float(cur_iter) / self.total_iters) ** self.lr_power)
            decayed_lr = self.min_lr + (self.start_lr - self.min_lr) * poly_decay
            self.curr_lr = max(self.min_lr, decayed_lr)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    optim = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0005, momentum=0.9)
    epochs = [100, 200]
    wu = 2
    lrsched = WarmUpPolyLR(optim, 0.0005, 0.9, 100 * 1000, wu * 1000)
    lrs = []
    for e in range(100):
        for step in range(1000):
            lrsched.update_lr(e * 1000 + step)
            lrs.append(lrsched.curr_lr)
    plt.plot(lrs)

    lrsched = WarmUpPolyLR(optim, 0.005, 0.9, 100 * 1000, wu * 1000)
    lrs = []
    for e in range(100):
        for step in range(1000):
            lrsched.update_lr(e * 1000 + step)
            lrs.append(lrsched.curr_lr)
    plt.plot(lrs)

    lrsched = WarmUpPolyLR(optim, 0.005, 2, 200 * 1000, 10 * 1000)
    lrs = []
    for e in range(200):
        for step in range(1000):
            lrsched.update_lr(e * 1000 + step)
            lrs.append(lrsched.curr_lr)
    plt.plot(lrs)

    plt.show()
