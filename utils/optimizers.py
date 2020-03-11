import torch

class Adam():
    def __init__(self, learning_rate= 1e-4, b1= 0.9, b2= 0.999):
        self.learning_rate= learning_rate
        self.eps= 1e-8
        self.m= None
        self.v= None
        # Decay rates
        self.b1= b1
        self.b2= b2
        
    def step(self, w, grad_wrt_w):
        # If not initialized
        if self.m is None:
            self.m= torch.zeros(grad_wrt_w.shape)
            self.v= torch.zeros(grad_wrt_w.shape)
            
        self.m= self.b1 * self.m + (1 -self.b1) * grad_wrt_w
        self.v= self.b2 * self.v + (1 -self.b2) * torch.pow(grad_wrt_w, 2)
        
        m_hat= self.m / (1 -self.b1)
        v_hat= self.v / (1 -self.b2)
        
        self.w_update= self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.eps)
        
        return w -self.w_update