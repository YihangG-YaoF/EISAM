import numpy as np
import torch

def mixup(batch, alpha):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    data, targets = batch
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(len(data))

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b, lam)

class MixUpCollator:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = mixup(batch, self.alpha)
        return batch
    
class SoftCrossEntropyLoss:
    def __init__(self, reduction):
        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction)

    def __call__(self, preds, targets):
        device = preds.device
        targets1, targets2, lam = targets
        return lam * self.criterion(
            preds, targets1.to(device)) + (1 - lam) * self.criterion(preds, targets2.to(device))