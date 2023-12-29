import torch


class R2Meter:
    def __init__(self, target_mean):
        super(R2Meter, self).__init__()
        self.target_mean = target_mean
        self.reset()

    def reset(self):
        self.n = 0
        self.ressum = 0.0
        self.totsum = 0.0

    def add(self, output, target):
        if not torch.is_tensor(output) and not torch.is_tensor(target):
            output = torch.from_numpy(output)
            target = torch.from_numpy(target)
        self.n += output.numel()
        self.ressum += torch.sum((output - target) ** 2).item()
        self.totsum += torch.sum((target - self.target_mean) ** 2).item()

    def value(self):
        r2 = (1 - (self.ressum / self.totsum)) if self.n > 0 and self.totsum > 0 else 0.0

        return r2
