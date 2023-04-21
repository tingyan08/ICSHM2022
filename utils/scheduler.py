import math
import torch
import matplotlib.pyplot as plt
from typing import Optional
from torch.optim import Optimizer



class LearningRateScheduler(object):
    r"""
    Provides inteface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']



    def simulate(self, num_update):
        lr = []
        for i in range(num_update):
            lr.append(self.step())
        return lr


class ExponetialLRScheduler(LearningRateScheduler):

    def __init__(
            self,
            optimizer: Optimizer,
            init_lr: float,
            peak_lr: float,
            final_lr: float,
            warmup_steps: int,
            total_steps: int
    ) -> None:
        assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"
    

        super(ExponetialLRScheduler, self).__init__(optimizer, init_lr)
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        final_lr_scale = self.final_lr / self.peak_lr
        self.warmup_rate = self.peak_lr / self.warmup_steps
        self.decay_factor = -math.log(final_lr_scale) / (self.total_steps - self.warmup_steps)

        self.init_lr = init_lr
        self.update_steps = 0

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps
    
        return 1, self.update_steps - self.warmup_steps



    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        self.update_steps += 1
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.update_steps * self.warmup_rate
        elif stage == 1:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)

        return self.lr

if __name__ == "__main__":
    model = torch.nn.Sequential(torch.nn.Linear(100,1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    total_steps = 1000
    scheduler = ExponetialLRScheduler(optimizer, init_lr=1e-5, peak_lr= 0.001, 
                                        final_lr=1e-6, warmup_steps=100, total_steps=total_steps)
    lr = scheduler.simulate(total_steps)
    print(lr[-1])
    plt.plot(range(len(lr)), lr)
    plt.show()