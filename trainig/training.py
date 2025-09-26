'''
In this file we define a class for training neural networks.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt
from uq_methods.UQ import NN_UQ
from typing import Optional
import os



class Trainer:

    def __init__(self, model: nn.Module, device: str = 'cpu', optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None, lr=0.001, **kwargs):
        '''
        Initialize the Trainer class with a neural network model.

        Args:
            model (torch.nn.Module): The neural network model to be trained.
            device (str): The device to run the computations on ('cpu' or 'cuda').
            optimizer (torch.optim.Optimizer, optional): Optimizer for training. If None, Adam optimizer will be used.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. If None, StepLR scheduler will be used.
            lr (float): Learning rate for the optimizer.
            kwargs: Additional keyword arguments for optimizer and scheduler.
        '''
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.optimizer = optimizer if optimizer is not None else optim.Adam(self.model.parameters(), lr=lr, **kwargs)
        self.scheduler = scheduler if scheduler is not None else optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9, **kwargs)
        # As different models have different loss functions, we assume it is defined in the model
        try:
            self.criterion = model.criterion
        except AttributeError:
            raise ValueError("Model must have a 'criterion' attribute defining the loss function.")
        self.uq = NN_UQ(model, device=device)

    def train(self, x_train, y_train,x_val,y_val, epochs=1000, batch_size=32, verbose=True,validation_freq:int = 500, save_dir: Optional[str] = None, saving_interval: Optional[int] = None):
        '''
        Train the neural network model.

        Parameters:
        x_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training target data.
        x_val (torch.Tensor): Validation input data.
        y_val (torch.Tensor): Validation target data.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        verbose (bool): If True, print training progress.
        '''
        save_model = False
        # Validation for saving parameters
        if save_dir is not None and saving_interval is None:
            raise ValueError("If save_dir is provided, saving_interval must also be specified.")
        if saving_interval is not None and save_dir is None:
            raise ValueError("If saving_interval is provided, save_dir must also be specified.")
        if save_dir is not None and saving_interval is not None:
            model_name = type(self.model).__name__
            directory = os.path.join(save_dir, model_name)
            os.makedirs(directory, exist_ok=True)
            save_model = True
        # Move data to device
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        # Create DataLoader for batching
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        loss_epochs = []

        self.model.train()
        prange = trange(epochs, desc="Training Progress", disable=not verbose)
        # Training loop
        for epoch in prange:
            epoch_loss = 0.0
            for inputs, targets in dataloader:               
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                prange.set_postfix({'Batch Loss': loss.item()})
                
                epoch_loss += loss.item() * inputs.size(0)

            self.scheduler.step()
            
            epoch_loss /= len(dataloader.dataset)
            loss_epochs.append(epoch_loss)
            if verbose and (epoch % validation_freq == 0 or epoch == epochs - 1):
                val_metrics,mean,std = self.uq.compute_uncertainty(x_val, y_val, n_samples=50)
                # prange.set_postfix({'Epoch Loss': epoch_loss, 'val metrics': val_metrics})
                print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.6f}, validation metrics: {val_metrics}")
            
            if save_model and (epoch % saving_interval == 0 or epoch == epochs - 1):
                save_path = os.path.join(directory, f"{model_name}_epoch{epoch+1}.pth")
                torch.save(self.model.state_dict(), save_path)
                if verbose:
                    print(f"Model saved to {save_path}")

        return losses, loss_epochs

    def plot_losses(self, losses, loss_epochs): 
        '''
        Plot training losses.

        Parameters:
        losses (list): List of batch-wise losses.
        loss_epochs (list): List of epoch-wise losses.
        '''
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Batch-wise Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss (Batch-wise)')
        plt.yscale('log')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(loss_epochs, label='Epoch-wise Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss (Epoch-wise)')
        plt.yscale('log')
        plt.legend()

        plt.tight_layout()
        plt.show()
