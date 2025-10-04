from torch import nn
import torch.nn.functional as F


class CEloss(nn.Module):
    def __init__(self):
        super(CEloss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)    
