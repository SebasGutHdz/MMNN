'''
In this file we create a class for Uncertainty Quantification (UQ) 
using trained neural networks.
'''

import torch 
import torch.nn as nn
import numpy as np




class NN_UQ:

    def __init__(self, model, device='cpu'):
        '''
        Initialize the UQ class with a trained neural network model.

        Parameters:
        model (torch.nn.Module): The trained neural network model.
        device (str): The device to run the computations on ('cpu' or 'cuda').
        '''
        self.model = model.to(device)
        self.device = device
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, x):
        '''
        Predict the output for given input x using the trained model.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Predicted output tensor.
        '''
        x = x.to(self.device)
        with torch.no_grad():
            y_pred = self.model(x)
        return y_pred.cpu()

    def compute_uncertainty(self, x, n_samples=100, confidence_level=0.95):
        '''
        Compute uncertainty in predictions using Monte Carlo.

        Parameters:
        x (torch.Tensor): Input tensor.
        n_samples (int): Number of stochastic forward passes.

        Returns:
        tuple: Mean and standard deviation of predictions.
        '''
        self.model.train()  # TODO: Identify type of model and set to train if MC dropout or BNN
        x = x.to(self.device)
        
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.model(x).cpu().numpy())
        
        preds = np.array(preds)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)

        # Compute confidence intervals (assuming Gaussian)
        z_scores = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z_score = z_scores.get(confidence_level, 1.96)

        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        self.model.eval()  # Set back to evaluation mode

        return mean_pred, std_pred, lower, upper

    
