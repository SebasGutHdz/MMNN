import torch
import torch.nn as nn



class MCDropoutNN(nn.Module):
    """
    Monte Carlo Dropout Neural Network
    Much more stable than full Bayesian approach!
    """
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=3, dropout_rate=0.1,n_dropout_layers= None):
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
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """Get predictions with uncertainty via MC Dropout"""
        self.train()  # Keep in training mode for dropout!
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        pred_mean = predictions.mean(dim=0)
        pred_std = predictions.std(dim=0)
        
        return pred_mean, pred_std