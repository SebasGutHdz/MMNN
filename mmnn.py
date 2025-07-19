import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class MMNN(nn.Module):
    def __init__(self, 
                 ranks = [1] + [16]*5 + [1], 
                 widths = [366]*6,
                 device = "cpu", 
                 ResNet = False,
                 fixWb = True):
        super().__init__()
        """
        A class to configure the neural network model.
    
        Attributes:
            ranks (list[int]): A list where the i-th element represents the output dimension of the i-th layer.
                               For the j-th layer, ranks[j-1] is the input dimension and ranks[j] is the output dimension.
            
            widths (list[int]): A list where each element specifies the width of the corresponding layer.
            
            device (str): The device (CPU/GPU) on which the PyTorch code will be executed.
            
            ResNet (bool): Indicates whether to use ResNet architecture, which includes identity connections between layers.
            
            fixWb (bool): If True, the weights and biases are not updated during training.
        """
        
        self.ranks = ranks
        self.widths = widths
        self.ResNet = ResNet
        self.depth = len(widths)
        
        fc_sizes = [ ranks[0] ] 
        for j in range(self.depth):
            fc_sizes += [ widths[j], ranks[j+1] ]

        fcs=[]
        for j in range(len(fc_sizes)-1):
            fc = nn.Linear(fc_sizes[j],
                           fc_sizes[j+1], device=device) 
            # setattr(self, f"fc{j}", fc)
            fcs.append(fc)
        self.fcs = nn.ModuleList(fcs)
        
        if fixWb:
            for j in range(len(fcs)):
                if j % 2 == 0:
                    self.fcs[j].weight.requires_grad = False
                    self.fcs[j].bias.requires_grad = False
                


    def forward(self, x):
        for j in range(self.depth):
            if self.ResNet:
                if 0 < j < self.depth-1:
                    x_id = x + 0
            x = self.fcs[2*j](x)
            x = torch.relu(x)
            x = self.fcs[2*j+1](x) 
            if self.ResNet:
                if 0 < j < self.depth-1:
                    n = min(x.shape[1], x_id.shape[1])
                    x[:,:n] = x[:,:n] + x_id[:,:n]
        return x



class MMNN_UQ(nn.Module):
    def __init__(self, 
                 ranks = [1] + [16]*5 + [1], 
                 widths = [366]*6,
                 device = "cpu", 
                 sigma = 0.01,
                 ResNet = False,
                 fixWb = True):
        super().__init__()
        """
        A class to configure the neural network model.
    
        Attributes:
            ranks (list[int]): A list where the i-th element represents the output dimension of the i-th layer.
                               For the j-th layer, ranks[j-1] is the input dimension and ranks[j] is the output dimension.
            
            widths (list[int]): A list where each element specifies the width of the corresponding layer.
            
            device (str): The device (CPU/GPU) on which the PyTorch code will be executed.
            
            ResNet (bool): Indicates whether to use ResNet architecture, which includes identity connections between layers.
            
            fixWb (bool): If True, the weights and biases are not updated during training.
        """
        
        self.ranks = ranks
        self.widths = widths
        self.ResNet = ResNet
        self.depth = len(widths)
        
        fc_sizes = [ ranks[0] ] 
        for j in range(self.depth):
            fc_sizes += [ widths[j], ranks[j+1] ]

        fcs=[]
        for j in range(len(fc_sizes)-1):
            fc = nn.Linear(fc_sizes[j],
                           fc_sizes[j+1], device=device) 
            
            # setattr(self, f"fc{j}", fc)
            fcs.append(fc)
        self.fcs = nn.ModuleList(fcs)
        
        if fixWb:
            for j in range(len(fcs)):
                if j % 2 == 0:
                    self.fcs[j].weight.requires_grad = False
                    self.fcs[j].bias.requires_grad = False
                    # For the last fixed layer
                    # if j == len(fcs)-2:
                    # Fix the first layer
                    if j == 0:
                        # print(f"Setting fixed weights and biases for layer {j} with sigma={sigma}")
                        self.sampler_W = MultivariateNormal(
                            loc = self.fcs[j].weight.view(-1),  
                            covariance_matrix = sigma**2 * torch.eye(self.fcs[j].weight.numel(), device=device)
                        )
                        self.sampler_b = MultivariateNormal(
                            loc = self.fcs[j].bias.view(-1),  
                            covariance_matrix = sigma**2 * torch.eye(self.fcs[j].bias.numel(), device=device)
                        )
                    
               
  

    def forward(self, x):
        for j in range(self.depth):
            if self.ResNet:
                if 0 < j < self.depth-1:
                    x_id = x + 0
            # Sample weights and biases for the last fixed layer and 
            # if 2*j == len(self.fcs)-2:
            # Sample weights and biases for the first fixed layer
            if 2*j == 0:
                weight_sample = self.sampler_W.sample().view(self.fcs[2*j].weight.shape)
                bias_sample = self.sampler_b.sample()
                x = torch.matmul(x, weight_sample.t()) + bias_sample
                # x = self.fcs[2*j](x)
            else:
                x = self.fcs[2*j](x)
            x = torch.relu(x)
            x = self.fcs[2*j+1](x) 
            if self.ResNet:
                if 0 < j < self.depth-1:
                    n = min(x.shape[1], x_id.shape[1])
                    x[:,:n] = x[:,:n] + x_id[:,:n]
        return x


