import torch
from utils.data_operations import accuracy_score

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()
    
    def gradient(self, y, y_pred):
        return NotImplementedError()
    
    def acc(self, y, y_pred):
        return 0
    
class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = torch.clamp(p, 1e-15, 1 - 1e-15)
        return - y * torch.log(p) - (1 - y) * torch.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(torch.argmax(y, axis=1), torch.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = torch.clamp(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)