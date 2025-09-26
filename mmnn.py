import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from typing import Optional, Callable

class SinTu(nn.Module):
    def __init__(self, s = -torch.pi):
        super().__init__()
        self.s = s

    def forward(self, x):
        max_x_s = torch.max(x, self.s*torch.ones_like(x))
        return torch.sin(max_x_s)

class MMNN(nn.Module):
    def __init__(self, 
                 ranks = [1] + [16]*5 + [1], 
                 widths = [366]*6,
                 device = "cpu", 
                 ResNet = False,
                 fixWb = True,
                 init_scaling = True,
                 activation: Optional[Callable] = None):
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
        if activation is None:
            self.activation = SinTu()
        else:
            self.activation = activation
        if len(ranks) != self.depth + 1:
            raise ValueError("Ranks needs the format [input_rank] + [hidden_ranks]*(depth-1) + [output_rank], and widths needs the format [width]*depth")
        
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
                if j==0 and init_scaling:
                    s = widths[0]/2
                    s = s**(1/ranks[0])*ranks[0]**0.5
                    self.fcs[j].weight.data = s * self.fcs[j].weight.data
                    self.fcs[j].bias.data = s * self.fcs[j].bias.data
                


    def forward(self, x):
        for j in range(self.depth):
            if self.ResNet:
                if 0 < j < self.depth-1:
                    x_id = x + 0
            x = self.fcs[2*j](x)
            x = self.activation(x)
            x = self.fcs[2*j+1](x) 
            if self.ResNet:
                if 0 < j < self.depth-1:
                    n = min(x.shape[1], x_id.shape[1])
                    x[:,:n] = x[:,:n] + x_id[:,:n]
        return x




class G_MMNN(nn.Module):
    def __init__(self, 
                 ranks = [1] + [16]*5 + [1], 
                 widths = [366]*6,
                 device = "cpu", 
                 sigma = 0.01,
                 n_gaussian_layers = None,
                 ResNet = False,
                 fixWb = True,
                 init_scaling = True,
                 activation: Optional[Callable] = None):
        super().__init__()
        
        """
        A class to configure the neural network model with Gaussian perturbation for UQ.
    
        Attributes:
            ranks (list[int]): A list where the i-th element represents the output dimension of the i-th layer.
                               For the j-th layer, ranks[j-1] is the input dimension and ranks[j] is the output dimension.
            
            widths (list[int]): A list where each element specifies the width of the corresponding layer.
            
            device (str): The device (CPU/GPU) on which the PyTorch code will be executed.
            
            sigma (float): Standard deviation for Gaussian perturbations.
            
            n_gaussian_layers (int): Number of W,b layers to perturb (from last to first).
                                   If None, all layers are perturbed.
                                   If 0, no layers are perturbed (deterministic).
            
            ResNet (bool): Indicates whether to use ResNet architecture.
            
            fixWb (bool): If True, the weights and biases are not updated during training.
        """
        
        self.ranks = ranks
        self.widths = widths
        self.ResNet = ResNet
        self.depth = len(widths)
        self.device = device
        self.sigma = sigma
        if activation is None:
            self.activation = SinTu()
        else:
            self.activation = activation
        if len(ranks) != self.depth + 1:
            raise ValueError("Ranks needs the format [input_rank] + [hidden_ranks]*(depth-1) + [output_rank], and widths needs the format [width]*depth")
        
        # Determine which layers should have Gaussian perturbations
        if n_gaussian_layers is None:
            self.n_gaussian_layers = self.depth
        else:
            self.n_gaussian_layers = max(0, min(n_gaussian_layers, self.depth))
        
        # Calculate which layer indices should be Gaussian (from last to first)
        if self.n_gaussian_layers > 0:
            start_idx = self.depth - self.n_gaussian_layers
            self.gaussian_layer_indices = set(range(start_idx, self.depth))
        else:
            self.gaussian_layer_indices = set()

        print(f"MMNN_UQ Configuration:")
        print(f"  Total W,b layers: {self.depth}")
        print(f"  Gaussian W,b layers: {self.n_gaussian_layers}")
        print(f"  Gaussian layer indices: {sorted(self.gaussian_layer_indices)}")

        # Build fc_sizes: [rank0, width0, rank1, width1, rank2, ..., rank_n]
        fc_sizes = [ranks[0]]
        for j in range(self.depth):
            fc_sizes += [widths[j], ranks[j+1]]

        # Initialize layers
        fcs = []
        for j in range(len(fc_sizes)-1):
            fc = nn.Linear(fc_sizes[j], fc_sizes[j+1], device=device) 
            fcs.append(fc)
        self.fcs = nn.ModuleList(fcs)
        
        # Initialize samplers dictionary
        self.samplers = {}
        
        if fixWb:
            for j in range(len(fcs)):
                if j % 2 == 0:  # W,b layers (even indices)
                    layer_idx = j // 2
                    
                    # Fix weights and biases for all W,b layers
                    self.fcs[j].weight.requires_grad = False
                    self.fcs[j].bias.requires_grad = False
                    if j == 0 and init_scaling :
                        s = widths[0]/2
                        s = s**(1/ranks[0])*ranks[0]**0.5
                        self.fcs[j].weight.data = s * self.fcs[j].weight.data
                        self.fcs[j].bias.data = s * self.fcs[j].bias.data
                    # Create samplers only for Gaussian layers
                    if layer_idx in self.gaussian_layer_indices:
                        print(f"  Creating Gaussian sampler for layer {layer_idx} (fcs[{j}])")
                        
                        # Create MultivariateNormal distributions for this specific layer
                        weight_flat = self.fcs[j].weight.view(-1)
                        bias_flat = self.fcs[j].bias.view(-1)
                        
                        weight_cov = sigma**2 * torch.eye(weight_flat.numel(), device=device)
                        bias_cov = sigma**2 * torch.eye(bias_flat.numel(), device=device)
                        
                        self.samplers[layer_idx] = {
                            'W': MultivariateNormal(
                                loc=weight_flat,  
                                covariance_matrix=weight_cov
                            ),
                            'b': MultivariateNormal(
                                loc=bias_flat,  
                                covariance_matrix=bias_cov
                            ),
                            'weight_shape': self.fcs[j].weight.shape,
                            'bias_shape': self.fcs[j].bias.shape
                        }
                
        print(f"  Total samplers created: {len(self.samplers)}")
                    
    
    
    
    def forward(self, x):
        """Forward pass with Gaussian perturbations on selected layers."""
        for j in range(self.depth):
            if self.ResNet:
                if 0 < j < self.depth-1:
                    x_id = x + 0
            
            # Apply W,b transformation (with or without Gaussian perturbation)
            if j in self.gaussian_layer_indices:
                # Sample perturbed weights and biases for this specific layer
                weight_sample = self.samplers[j]['W'].sample().view(self.samplers[j]['weight_shape'])
                bias_sample = self.samplers[j]['b'].sample().view(self.samplers[j]['bias_shape'])
                x = torch.matmul(x, weight_sample.t()) + bias_sample
            else:
                # Use deterministic weights
                x = self.fcs[2*j](x)
            
            # Apply activation
            x = self.activation(x)
            
            # Apply A,c transformation (always deterministic and trainable)
            x = self.fcs[2*j+1](x) 
            
            if self.ResNet:
                if 0 < j < self.depth-1:
                    n = min(x.shape[1], x_id.shape[1])
                    x[:,:n] = x[:,:n] + x_id[:,:n]
        return x
    
    