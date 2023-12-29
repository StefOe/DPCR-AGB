import torch


class MAEMeter:
    def __init__(self):
        super(MAEMeter, self).__init__()
        self.reset()

    def reset(self):
        self.n = 0
        self.abssum = 0.0

    def add(self, output, target):
        if not torch.is_tensor(output) and not torch.is_tensor(target):
            output = torch.from_numpy(output)
            target = torch.from_numpy(target)
        self.n += output.numel()
        self.abssum += torch.sum(abs(output - target)).item()

    def value(self):
        mae = self.abssum / max(1, self.n)
        return mae
