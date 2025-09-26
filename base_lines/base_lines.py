import torch
import torch.nn as nn



class MCDropoutNN(nn.Module):
    """
    Monte Carlo Dropout Neural Network
    """
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=3, dropout_rate=0.1,n_dropout_layers= None):
        '''
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in each hidden layer.
            output_size (int): Number of output features.
            num_layers (int): Total number of layers (including input and output layers).
            dropout_rate (float): Dropout rate (between 0 and 1).
            n_dropout_layers (int): Number of dropout layers to include before the output layer. If None, defaults to num_layers - 1.
        '''
        super().__init__()
        
        self.dropout_rate = dropout_rate
        layers = []
        if n_dropout_layers is None:
            n_dropout_layers = num_layers - 1
        dropout_layers = []
        # Build architecture
        layer_sizes = [input_size] + [hidden_size] * (num_layers-1) + [output_size]
        
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:  # No activation on last layer
                layers.append(nn.Tanh())
                if i>= len(layer_sizes)-2 - n_dropout_layers:
                    dropout_layers.append(i)  # Store dropout layer positions
                    layers.append(nn.Dropout(dropout_rate))
        print(f"Using dropout in layers: {dropout_layers}")
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()
    