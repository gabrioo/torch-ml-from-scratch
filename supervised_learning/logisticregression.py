import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from utils.activation_functions import Sigmoid
from utils.data_operations import make_diagonal

class LogisticRegression(nn.Module):
    
    def __init__(self, learning_rate= 0.001, gradient_descent= True):
        super(LogisticRegression, self).__init__()
        self.param= None
        self.learning_rate= learning_rate
        self.gradient_descent= gradient_descent
        self.sigmoid= Sigmoid()

    def _initialize_parameters(self, X):
        n_features= X.shape[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit= 1/math.sqrt(n_features)
        
        # self.param= torch.distributions.Uniform(-limit, limit, (n_features,))
        
        distribution = torch.distributions.Uniform(torch.Tensor([-limit]), torch.Tensor([limit]))
        self.param= distribution.sample(torch.Size([n_features])).double()

    def fit(self, X, y, n_iterations= 4000):
        self._initialize_parameters(X)
        for _ in range(n_iterations):
            dot_product= torch.mm(X, self.param)
            y_pred= self.sigmoid(dot_product)
            if self.gradient_descent:
                y_pred= y_pred.view(y_pred.shape[0])
                diff= -(y.double() -y_pred)
                diff= diff.view(1, diff.shape[0])
                self.param-= self.learning_rate * torch.transpose(torch.mm(diff, X), 0, 1)
            else:
                diag_gradient= make_diagonal(self.sigmoid.gradient(X.dot(self.param)))
                self.param= torch.inverse(X.T.dot(diag_gradient)).Int()

    def predict(self, X):
        mm_product= torch.mm(X, self.param).tolist()
        mm_product= torch.Tensor([i[0] for i in mm_product])
        y_pred= torch.round(self.sigmoid(mm_product)).int()
        return y_pred