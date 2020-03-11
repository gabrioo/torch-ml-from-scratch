import torch

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def gradient(self, x):
        grd= self.__call__(x) * (1 -self.__call__(x))
        return grd
    
class Softmax():
    def __call__(self, x):
        max_x= torch.max(x, dim=-1, keepdims=True)
        e_x = torch.exp(x - max_x[0])
        return e_x / torch.sum(e_x, dim=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)