import torch
import torch.nn as nn
import math

from utils.activation_functions import Sigmoid, Softmax
from utils.loss_functions import CrossEntropy

class MultiLayerPerceptron(nn.Module):
    
    def __init__(self, n_hidden, n_iterations= 3000, learning_rate= 1e-4):
        super(MultiLayerPerceptron, self).__init__()
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()
        
    def _initialize_weights(self, X, y):
        _, n_features= X.shape
        _, n_outputs= y.shape
        
        # Hidden Layer
        limit= 1 / math.sqrt(n_features)
        distribution = torch.distributions.Uniform(torch.Tensor([-limit]), torch.Tensor([limit]))
        self.W= distribution.sample(torch.Size([n_features, self.n_hidden])).double().view(n_features, self.n_hidden)
        self.w0= torch.zeros((1, self.n_hidden)).double()
        
        # Output Layer
        limit= 1 / math.sqrt(n_outputs)
        distribution = torch.distributions.Uniform(torch.Tensor([-limit]), torch.Tensor([limit]))
        self.V= distribution.sample(torch.Size([self.n_hidden, n_outputs])).double().view(self.n_hidden, n_outputs)
        self.v0= torch.zeros((1, n_outputs)).double()
        
    def fit(self, X, y):
        self._initialize_weights(X, y)
        
        for i in range(self.n_iterations):
            
            # Forward pass
            hidden_input= torch.mm(X, self.W) + self.w0
            hidden_output= self.hidden_activation(hidden_input)
            output_layer_input= torch.mm(hidden_output, self.V) + self.v0
            y_pred= self.output_activation(output_layer_input)
            
            # Backward pass
            
            # Gradients w.r.t input of the output layer
            grad_wrt_out_1_input= self.loss.gradient(y, y_pred) * self.output_activation.gradient(output_layer_input)
            grad_v= torch.mm(hidden_output.t(), grad_wrt_out_1_input)
            grad_v0= torch.sum(grad_wrt_out_1_input, dim= 0, keepdim= True)
            
            # Gradients w.r.t input of hidden layer
            grad_wrt_hidden_1_input= torch.mm(grad_wrt_out_1_input, self.V.t()) * self.hidden_activation.gradient(hidden_input)
            grad_w= torch.mm(X.t(), grad_wrt_hidden_1_input)
            grad_w0= torch.sum(grad_wrt_hidden_1_input, dim= 0, keepdim= True)
            
            # print(grad_v.shape, self.V.shape)
            # print(grad_v0.shape, self.v0.shape)
            # print(grad_w.shape, self.W.shape)
            # print(grad_w0.shape, self.w0.shape)
            
            self.V -= self.learning_rate * grad_v
            self.v0 -= self.learning_rate * grad_v0
            self.W -= self.learning_rate * grad_w
            self.w0 -= self.learning_rate * grad_w0
            
    def predict(self, X):
        # Forward pass
        hidden_input= torch.mm(X, self.W) + self.w0
        hidden_output= self.hidden_activation(hidden_input)
        output_layer_input= torch.mm(hidden_output, self.V) + self.v0
        y_pred= self.output_activation(output_layer_input)
        return y_pred