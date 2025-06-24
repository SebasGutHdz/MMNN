import jax
import jax.random as jrandom
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Sequence, Callable, Optional, Dict, Any
import numpy as np


class SinActivation(nn.Module):
    """Sine activation function."""
    
    def __call__(self, x):
        return jnp.sin(x)


class SinTUActivation(nn.Module):
    """Sine Truncated Unit activation function: SinTU_s = sin(max(x, s))"""
    s: float = -jnp.pi  # Default truncation parameter
    
    def __call__(self, x):
        return jnp.sin(jnp.maximum(x, self.s))
    
class MMNNLayer(nn.Module):
    """
    MMNN Layer with proper setup initialization
    
    Args:
        d_in: Input dimension
        width: Width of the hidden layer (number of random basis functions)
        d_out: Output dimension
        activation: Activation function to apply
        use_bias: Whether to use bias in the fixed layer
        seed: Random seed for reproducible fixed weights
    """
    d_in: int
    width: int 
    d_out: int
    activation: Callable = SinTUActivation()
    use_bias: bool = True
    seed: int = 0
    
    def setup(self):
        """Initialize fixed W and b matrices during setup"""
        # Create deterministic random key for fixed weights
        key = jrandom.PRNGKey(self.seed)
        key_w, key_b = jrandom.split(key)
        
        # Initialize fixed W matrix: maps from d_in to width
        # Using Xavier/Glorot initialization scaled by input dimension
        self.W = jrandom.normal(key_w, shape=(self.width, self.d_in)) * jnp.sqrt(2.0 / self.d_in)
        
        # Initialize fixed bias vector
        
        self.b = jrandom.normal(key_b, shape=(self.width,))* jnp.sqrt(2.0 / self.d_in)
        
        
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through MMNN layer
        
        Args:
            x: Input tensor of shape (..., d_in)
            
        Returns:
            Output tensor of shape (..., d_out)
        """
        # Verify input dimension
        if x.shape[-1] != self.d_in:
            raise ValueError(f"Expected input dimension {self.d_in}, got {x.shape[-1]}")
        
        # Apply fixed linear transformation
        hidden = jnp.dot(x, self.W.T)+self.b
    
        # Apply activation function
        activated = self.activation(hidden)
        
        # Apply trainable Dense layer: width -> d_out
        output = nn.Dense(
            features=self.d_out,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.glorot_uniform(),
        )(activated)
        
        return output

class MMNNModel(nn.Module):
    '''
    MMNN model
    Args:
        ranks: List of dimensions [d_in,d1,d2,...,d_out] of length n+1
        widhts : List of widths for each layer [w_1,w_2,...,w_n] of length n
        activation: Activation function to apply to each layer
        use_bias: Whether to use bias in the fixed layer
        seed: Random seed for reproducible fixed weights
    '''
    ranks: list
    widths: list
    activation: Callable = SinTUActivation()
    use_bias: bool = True
    seed: int = 0

    def create_layer_configs(self):
        '''
        Create layer configurations from ranks and widths

        Returns:
            List of (d_in, width,d_out) tuples for each layer
        
        '''
        if len(self.widths)+1 != len(self.ranks):
            raise ValueError("Number of widths must be one less than number of ranks")

        layer_configs = []
        for i in range(len(self.widths)):
            d_in = self.ranks[i]
            width = self.widths[i]
            d_out = self.ranks[i+1]
            layer_configs.append((d_in, width, d_out))
        return layer_configs
    
    def setup(self):
        """Setup all MMNN layers"""
        # Create layer configurations from ranks and widths
        layer_configs = self.create_layer_configs()
        
        # Create layers as individual attributes instead of a list
        # This is the proper way to handle multiple submodules in Flax
        for i, (d_in, width, d_out) in enumerate(layer_configs):
            layer = MMNNLayer(
                d_in=d_in,
                width=width,
                d_out=d_out,
                activation=self.activation,
                use_bias=self.use_bias,
                seed=self.seed + i,  # Different seed for each layer
            )
            # Set each layer as an attribute with a unique name
            setattr(self, f'layer_{i}', layer)
        
        # Store the number of layers for the forward pass
        self.num_layers = len(layer_configs)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through all layers"""
        for i in range(self.num_layers):
            layer = getattr(self, f'layer_{i}')
            x = layer(x)
        return x


class Train_jax_model():
    '''
    This a basic training scheme for a jax model.
    '''
    def __init__(self, model: nn.Module,
                 input_data: jnp.ndarray,
                 target_data: jnp.ndarray,
                 optimizer: str = 'adam',
                 loss_fn: str = 'mse',
                 learning_rate: float = 0.001,
                 num_epochs: int = 1000,
                 batch_size: int = 32,
                 random_seed: int = 0):
        self.model = model
        self.loss_fn = self._create_loss_fn(loss_fn)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed

        self.device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]

        input_data = jax.device_put(input_data, self.device)
        target_data = jax.device_put(target_data, self.device)


        self.n_samples = jnp.shape(input_data)[0]
        
        self.key = jrandom.PRNGKey(random_seed)
        self.split_train_test(input_data, target_data) # Stores self.x_train, self.y_train, self.x_test, self.y_test
        self.n_batches = jnp.shape(self.x_train)[0] // self.batch_size
        
        
        # Create optimizer
        if optimizer == 'adam':
            self.optimizer = optax.adam(learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = optax.sgd(learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Create training step function
        self.train_step = self._create_train_step()
        
    # Create loss function
    def _create_loss_fn(self, loss_type):
        if loss_type == 'mse':
            return lambda params, x, y: jnp.mean((self.model.apply(params, x) - y) ** 2)
        elif loss_type == 'mae':
            return lambda params, x, y: jnp.mean(jnp.abs(self.model.apply(params, x) - y))
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

    # Process input and target data
    def split_train_test(self,input_data,target_data,test_split = 0.2):
        """
        Split input and target data into training and validation sets
        """
        n_test = int(self.n_samples * test_split)
        n_train = self.n_samples - n_test

        # Random indices for shuffling
        indices = jrandom.permutation(self.key,self.n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        # Split input and target data
        self.x_train = input_data[train_indices]
        self.y_train = target_data[train_indices]
        self.x_test = input_data[test_indices]
        self.y_test = target_data[test_indices]

    # Batch generator
    def batch_generator(self, x_data, y_data):
        """
        Generate batches of data
        Inputs:
            x_data: Input data
            y_data: Target data
        Yields:
            x_batch: Input batch
            y_batch: Target batch
        """
        # The last incomplete batch will be ignored
        for i in range(0, self.n_batches * self.batch_size, self.batch_size):
            x_batch = x_data[i:i + self.batch_size]
            y_batch = y_data[i:i + self.batch_size]
            # yield is used to create a generator
            # This allows us to iterate over batches without loading everything into memory
            yield x_batch, y_batch
        
    # Define training step function 
    def _create_train_step(self):
        """Create a JIT-compiled training step function"""
        
        @jax.jit
        def train_step(params, opt_state, x_batch, y_batch):
            # Capture self variables in closure
            loss, grads = jax.value_and_grad(self.loss_fn)(params, x_batch, y_batch)
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        
        return train_step
    

    def training_loop(self,print_every: int = 100):

        import matplotlib.pyplot as plt
        """
        Training loop
        """
            
        training_losses = []
        validation_losses = []

        # Initialize model parameters
        sample_input = self.x_train[:1]  # Use first sample for initialization
        params = self.model.init(self.key, sample_input)
        # Move parameters to the device
        params = jax.device_put(params, self.device)
        self.params_store = params
        # Create optimizer state
        opt_state = self.optimizer.init(params)

        for epoch in range(self.num_epochs):

            epoch_loss = []
            self.key, subkey = jrandom.split(self.key)
            perm = jrandom.permutation(subkey, len(self.x_train))
            x_train_shuffled = self.x_train[perm]
            y_train_shuffled = self.y_train[perm]
            # Iterate over batches
            for x_batch, y_batch in self.batch_generator(x_train_shuffled, y_train_shuffled):

                # Perform a training step
                params, opt_state, loss = self.train_step(params, opt_state, x_batch, y_batch)
                epoch_loss.append(loss)

            self.params_store = params  # Store the latest parameters
            # Average loss for the epoch
            avg_loss = jnp.mean(jnp.array(epoch_loss)) 
            training_losses.append(avg_loss)

            

            if epoch % print_every == 0 or epoch == self.num_epochs - 1:
                # Compute validation loss
                val_loss = self.loss_fn(params, self.x_test, self.y_test)
                validation_losses.append(val_loss)
                # Plot the model predictions
                idx_sort = jnp.argsort(self.x_test, axis=0)  # Sort for consistent plotting
                x_test_local = self.x_test[idx_sort]
                y_test_local = self.y_test[idx_sort]
                predictions = self.model.apply(params, x_test_local)
                plt.figure(figsize=(10, 5))
                plt.plot(x_test_local.flatten(), y_test_local.flatten(), label='True Function', color='blue')
                plt.plot(x_test_local.flatten(), predictions.flatten(), label='Model Predictions', color='red')
                plt.title(f'Epoch {epoch + 1}/{self.num_epochs} - Validation Loss: {val_loss:.4f}')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                plt.grid()
                plt.show()
                print(f"Epoch {epoch + 1}/{self.num_epochs}, "
                      f"Training Loss: {avg_loss:.4f}, "
                      f"Validation Loss: {val_loss:.4f}")
        
                
        return params, {'training_losses': training_losses, 'validation_losses': validation_losses}


                    



