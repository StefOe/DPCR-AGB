import torch


class APPRXMeter:
    def __init__(self):
        super(APPRXMeter, self).__init__()
        self.reset()

    def reset(self):
        self.n = 0
        self.target_sum = 0.0
        self.output_sum = 0.0

    def add(self, output, target):
        if not torch.is_tensor(output) and not torch.is_tensor(target):
            output = torch.from_numpy(output)
            target = torch.from_numpy(target)
        self.n += output.numel()
        self.target_sum += torch.sum(target).item()
        self.output_sum += torch.sum(output).item()

    def value(self):
        apprx = abs(1 - self.output_sum/self.target_sum) if self.n > 0 else 0.0

        return apprx
