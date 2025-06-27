import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax
from typing import Callable, Tuple, List, Optional, Sequence
import matplotlib.pyplot as plt


# 1. Activation functions converted to Equinox Modules
class SinActivation(eqx.Module):
    def __call__(self, x):
        return jnp.sin(x)

class SinTUActivation(eqx.Module):
    s: float = -jnp.pi

    def __call__(self, x):
        return jnp.sin(jnp.maximum(x, self.s))


# 2. MMNNLayer now initializes all parameters in __init__
class MMNNLayer(eqx.Module):
    W: jnp.ndarray
    b: jnp.ndarray
    linear: eqx.nn.Linear
    activation: Callable

    def __init__(self, d_in: int, width: int, d_out: int, 
                 activation: Callable = SinTUActivation(),
                 use_bias: bool = True,
                 key: jrandom.PRNGKey = None):
        key_w, key_b, key_linear = jrandom.split(key, 3)
        
        scale = jnp.sqrt(2.0 / d_in)
        self.W = scale * jrandom.normal(key_w, (width, d_in))
        self.b = scale * jrandom.normal(key_b, (width,))
        
        # FIX: Corrected input dimensions for linear layer
        self.linear = eqx.nn.Linear(
            in_features=width,  # Input features should match layer width
            out_features=d_out,  # Output features should match d_out
            use_bias=use_bias,
            key=key_linear
        )
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        W_sg = jax.lax.stop_gradient(self.W)
        b_sg = jax.lax.stop_gradient(self.b)
        
        # FIX: Corrected matrix multiplication order
        # x shape: (batch, d_in)
        # W_sg shape: (width, d_in) -> transpose to (d_in, width) for matmul
        hidden = jnp.matmul(x, W_sg.T) + b_sg
        
        # Apply activation and trainable layer
        return self.linear(self.activation(hidden))

# 3. MMNNModel uses tuple storage for layers
class MMNNModel(eqx.Module):
    layers: Tuple[MMNNLayer, ...]
    num_layers: int

    def __init__(self, ranks: List[int], widths: List[int], 
                 activation: Callable = SinTUActivation(), 
                 use_bias: bool = True, 
                 key: jrandom.PRNGKey = None):
        # Validation
        if len(widths) + 1 != len(ranks):
            raise ValueError("Number of widths must be one less than number of ranks")
        
        # Create layer configurations
        layer_configs = []
        for i in range(len(widths)):
            layer_configs.append((ranks[i], widths[i], ranks[i+1]))
        
        # Split keys for each layer
        keys = jrandom.split(key, len(layer_configs)) if key is not None else [None]*len(layer_configs)
        
        # Initialize layers
        layers = []
        for i, (d_in, width, d_out) in enumerate(layer_configs):
            layer = MMNNLayer(
                d_in=d_in,
                width=width,
                d_out=d_out,
                activation=activation,
                use_bias=use_bias,
                key=keys[i]
            )
            layers.append(layer)
        
        # Store as tuple (Equinox requires PyTree-compatible storage)
        self.layers = tuple(layers)
        self.num_layers = len(layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Sequential forward pass
        for layer in self.layers:
            x = layer(x)
        return x


# 4. Training class updated for Equinox paradigm
class Train_Equinox_Model:
    def __init__(self, model: eqx.Module,
                 input_data: jnp.ndarray,
                 target_data: jnp.ndarray,
                 optimizer: str = 'adam',
                 loss_type: str = 'mse',
                 learning_rate: float = 0.001,
                 num_epochs: int = 1000,
                 batch_size: int = 32,
                 random_seed: int = 0):
        self.model = model
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.key = jrandom.PRNGKey(random_seed)
        
        # Device setup
        self.device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
        
        # Data handling
        self.n_samples = input_data.shape[0]
        self.split_train_test(input_data, target_data)
        
        # Optimizer setup
        if optimizer == 'adam':
            self.optimizer = optax.adam(learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = optax.sgd(learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Training step setup
        self.train_step = self._create_train_step()

    def split_train_test(self, input_data, target_data, test_split=0.2):
        n_test = int(self.n_samples * test_split)
        n_train = self.n_samples - n_test
        
        # Shuffle data
        key, subkey = jrandom.split(self.key)
        indices = jrandom.permutation(subkey, self.n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Move data to device
        self.x_train = jax.device_put(input_data[train_indices], self.device)
        self.y_train = jax.device_put(target_data[train_indices], self.device)
        self.x_test = jax.device_put(input_data[test_indices], self.device)
        self.y_test = jax.device_put(target_data[test_indices], self.device)
        
        # Calculate batches
        self.n_batches = n_train // self.batch_size

    def batch_generator(self, x_data, y_data):
        # Generate complete batches (drops last partial batch)
        for i in range(0, self.n_batches * self.batch_size, self.batch_size):
            yield x_data[i:i+self.batch_size], y_data[i:i+self.batch_size]

    def _create_train_step(self):
        # CHANGE: Uses Equinox's filtered transformations
        @eqx.filter_jit
        def train_step(model, opt_state, x_batch, y_batch, optimizer):
            # Define loss function
            def loss_fn(model):
                pred = model(x_batch)
                if self.loss_type == 'mse':
                    return jnp.mean((pred - y_batch) ** 2)
                elif self.loss_type == 'mae':
                    return jnp.mean(jnp.abs(pred - y_batch))
                else:
                    raise ValueError(f"Unsupported loss: {self.loss_type}")
            
            # Compute loss and gradients
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss
        
        return train_step

    def training_loop(self, print_every: int = 100):
        training_losses = []
        validation_losses = []
        
        # Initialize optimizer state
        # CHANGE: Only optimize array leaves
        opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_loss = []
            self.key, subkey = jrandom.split(self.key)
            perm = jrandom.permutation(subkey, len(self.x_train))
            x_train_shuffled = self.x_train[perm]
            y_train_shuffled = self.y_train[perm]
            
            for x_batch, y_batch in self.batch_generator(x_train_shuffled, y_train_shuffled):
                self.model, opt_state, loss = self.train_step(
                    self.model, opt_state, x_batch, y_batch, self.optimizer
                )
                print(f"Epoch {epoch+1}/{self.num_epochs}, Batch Loss: {loss:.4f}")
                epoch_loss.append(loss)
            
            avg_loss = jnp.mean(jnp.array(epoch_loss))
            training_losses.append(avg_loss)
            
            if epoch % print_every == 0 or epoch == self.num_epochs - 1:
                # Validation loss
                val_loss = self.loss_fn(self.model, self.x_test, self.y_test)
                validation_losses.append(val_loss)
                
                # Plotting
                idx_sort = jnp.argsort(self.x_test, axis=0)
                x_test_sorted = self.x_test[idx_sort]
                y_test_sorted = self.y_test[idx_sort]
                predictions = self.model(x_test_sorted)
                
                plt.figure(figsize=(10, 5))
                plt.plot(x_test_sorted.flatten(), y_test_sorted.flatten(), 
                         label='True Function', color='blue')
                plt.plot(x_test_sorted.flatten(), predictions.flatten(), 
                         label='Model Predictions', color='red')
                plt.title(f'Epoch {epoch+1}/{self.num_epochs} - Val Loss: {val_loss:.4f}')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                plt.grid()
                plt.show()
                
                print(f"Epoch {epoch+1}/{self.num_epochs}, "
                      f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return self.model, {
            'training_losses': training_losses,
            'validation_losses': validation_losses
        }

    def loss_fn(self, model, x, y):
        """Convenience method for external loss calculation"""
        pred = model(x)
        if self.loss_type == 'mse':
            return jnp.mean((pred - y) ** 2)
        elif self.loss_type == 'mae':
            return jnp.mean(jnp.abs(pred - y))
        raise ValueError(f"Unsupported loss: {self.loss_type}")