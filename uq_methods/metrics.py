''''
In this file we implement the metrics to evaluate the performance of the UQ methods.
The metrics are: RMSE, energy score (ES), continuous ranked probability score (CRPS), negative log-likelihood (NLL),
coverage of the predictive distribution for selected alpha quantile C_{\alpha}, and the 
prediction interval |C_{\alpha}|.
 
Rerence: PROBABILISTIC NEURAL OPERATORS FOR FUNCTIONAL  UNCERTAINTY QUANTIFICATION (arxiv.2502.12902v2) 
eqns 11-16
'''


import torch 
from typing import Union,Tuple
import torch.nn as nn
import numpy as np


class Metrics:

    '''UQ metric for NN predictions. Each method takes as input a prediction sample and the ground truth.'''

    def __init__(self):
        pass

    def l2_loss(self, pred_mean: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''Compute the RMSE between the prediction mean and the ground truth'''
        return torch.sqrt(nn.MSELoss(reduction='mean')(pred_mean, y))
    
    def energy_score(self, pred_samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''
        Compute the energy score (ES) between the prediction samples and the ground truth
        Args:
            pred_samples: (S, D) tensor, prediction samples.  S: samples per input, D: output dimension
            y: (D) tensor, ground truth
        Returns:
            es: scalar tensor, energy score        
        '''
        S, D = pred_samples.shape
        
        # Term 1: mean distance to truth
        term1 = torch.norm(pred_samples - y, dim=-1).mean()  # scalar
        
        # Term 2: mean pairwise distance (excluding diagonal)
        pairwise_dists = torch.norm(
            pred_samples.unsqueeze(0) - pred_samples.unsqueeze(1), 
            dim=-1
        )  # (S, S)
        
        # Remove diagonal (m=h terms) and normalize correctly
        mask = ~torch.eye(S, dtype=torch.bool, device=pred_samples.device)
        term2 = pairwise_dists[mask].mean() / 2  # scalar
        
        return term1 - term2  # scalar
    
    def crps(self, pred_samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Continuous Ranked Probability Score (CRPS) using quantile scoring.
        Args:
            pred_samples: (S, D) tensor, prediction samples 
            y: (D) tensor, ground truth
        Returns:
            crps: scalar tensor, CRPS score
        """
        def quantile_score(y_true, y_pred_quantile, alpha):
            indicator = (y_true < y_pred_quantile).float()
            return (alpha - indicator) * (y_true - y_pred_quantile)

        alphas = torch.linspace(0.01, 0.99, 99, device=pred_samples.device)
        
        # Vectorized quantile computation
        quantiles = torch.quantile(pred_samples, alphas, dim=0)  # (99, D)
        
        # Compute quantile scores for all alphas at once
        y_expanded = y.unsqueeze(0)  # (1, D)
        alphas_expanded = alphas.unsqueeze(1)  # (99, 1)
        
        qs = quantile_score(y_expanded, quantiles, alphas_expanded)  # (99, D)
        qs_alpha = qs.mean(dim=1)  # (99,) - average over D dimension
        
        crps = torch.trapezoid(qs_alpha, alphas)
        return crps   
    
    def nll(self, pred_mean: torch.Tensor, pred_var: torch.Tensor, y: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
        '''
        Compute the Negative Log-Likelihood (NLL) assuming a Gaussian distribution for the predictions.
        Args:
            pred_mean: (D) tensor, prediction mean
            pred_var: (D) tensor, prediction variance
            y: (D) tensor, ground truth
            eps: small value to avoid log(0)
        Returns:
            nll: scalar tensor, NLL score
        ''' 
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=pred_mean.device))
        return 0.5 * torch.mean(torch.log(pred_var + eps) + (y - pred_mean) ** 2 / (pred_var + eps)) + 0.5 * log_2pi
        
    
    def coverage(self, pred_samples: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Compute coverage and interval size.
        Args:
            pred_samples: (S, D) tensor, prediction samples
            y: (D) tensor, ground truth  
            alpha: float, quantile level for prediction interval
        Returns:
            coverage: scalar tensor, coverage probability
            interval_size: scalar tensor, average interval width
        '''
        lower_quantile = torch.quantile(pred_samples, alpha / 2, dim=0)  # (D,)
        upper_quantile = torch.quantile(pred_samples, 1 - alpha / 2, dim=0)  # (D,)
        
        # Fix: Add parentheses around the boolean operation
        coverage = torch.mean(((y >= lower_quantile) & (y <= upper_quantile)).float())  # scalar
        interval_size = torch.mean(torch.abs(upper_quantile - lower_quantile))  # scalar
        
        return coverage, interval_size


        
    





