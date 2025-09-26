'''
In this file we create a class for Uncertainty Quantification (UQ) 
using trained neural networks.
'''
import os
import sys
path_ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path_ not in sys.path:
    sys.path.append(path_)

import torch 
import torch.nn as nn
import numpy as np
from uq_methods.metrics import Metrics




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
        self.metrics = Metrics()

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

    def compute_uncertainty(self, x: torch.Tensor, y: torch.Tensor, n_samples: int =100, confidence_level: float =0.95):
        '''
        Compute uncertainty in predictions using Monte Carlo.

        Args:
            x : Input tensor of the domain (N,D_in)
            y : Ground truth tensor (N,D_out)
            n_samples (int): Number of stochastic forward passes.

        Returns:
            dictionary with Metrics: l2_loss, es, crps, nll, coverage, pred_interval
        '''
        # MC Dropout require the model to be in training mode for stochasticity
        if self.model.__class__.__name__ in ['MC_Dropout']:
            self.model.train() 
        x = x.to(self.device)
        # Perform multiple stochastic forward passes
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.model(x))
        
        preds = torch.stack(preds) # (n_samples,N, D_out)
        mean_pred = preds.mean(axis=0) # (N, D_out)
        std_pred = preds.std(axis=0) # (N, D_out)
        preds_batched = preds.transpose(0, 1)  # (N, n_samples, D_out)
        
        # Collect uq metrics
        l2_loss = self.metrics.l2_loss(mean_pred, y)
        # For functions returning single values
        crps_scores = torch.vmap(self.metrics.crps)(preds_batched, y).mean()
        nll_scores = torch.vmap(self.metrics.nll)(mean_pred, std_pred, y).mean()
        es_scores = torch.vmap(self.metrics.energy_score)(preds_batched, y).mean()

        # For functions returning tuples
        coverage_results = torch.vmap(lambda pred, y: self.metrics.coverage(pred, y, alpha=0.1))(preds_batched, y)
        coverage_probs, interval_sizes = coverage_results  # Unpack the tuple
        coverage_probs = coverage_probs.mean()
        interval_sizes = interval_sizes.mean()

        results = {
            'l2_loss': l2_loss.item(),
            'crps': crps_scores.item(),
            'nll': nll_scores.item(),
            'es': es_scores.item(),
            'coverage': coverage_probs.item(),
            'pred_interval': interval_sizes.item()
        }


        return results,mean_pred, std_pred