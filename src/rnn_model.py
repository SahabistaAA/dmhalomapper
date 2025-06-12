import torch as tc
import torch.nn as tcn
import torch.nn.functional as F
from torchinfo import summary
from directory_manager import PathManager

path_manager = PathManager()
logger = path_manager.get_logger()

# Base class for RNN Implementation
class DMHaloMatterRNN(tcn.Module):
    """
    Base class for RNN architecture.
    
    Args:
        input_size (int): Input size.
        hidden_size (int): Hidden size.
        output_size (int): Output size.
        num_layers (int): Number of layers. Defaults to 2.
        dropout_rate (float): Dropout rate. Defaults to 0.2.
        activation (str): Activation function. Defaults to 'gelu'.
        bidirectional (bool): Whether to use bidirectional RNN. Defaults to False.
        normalize_input (bool): Whether to normalize input. Defaults to False.
    
    Attrs:
        batch_norm (tcn.BatchNorm1d): Batch normalization layer.
        feature_layers (tcn.Sequential): Feature extraction layers before RNN.
        output_layer (tcn.Sequential): Output projection layers after RNN.
    """
    def __init__(self, 
                    input_size=None, 
                    hidden_size=None, 
                    output_size=None, 
                    num_layers=2, 
                    dropout_rate=0.2,
                    activation='gelu', 
                    bidirectional=False, 
                    feature_dim=None,
                    use_layer_norm=False, 
                    return_hidden=False):
        super().__init__()
        self.input_size = input_size if input_size is not None else 64
        self.hidden_size = hidden_size if hidden_size is not None else 128
        self.output_size = output_size if output_size is not None else 1
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.activation = self.__get_activation_layer(activation)
        self.bidirectional = bidirectional
        self.feature_dim = feature_dim or min(self.hidden_size, self.input_size)
        self.use_layer_norm = use_layer_norm
        self.return_hidden = return_hidden
        self.device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
        self.normalize_input = None
        
        # Log initialization
        print(f"Initializing DMHaloMatterRNN with input_size={self.input_size}, "
                     f"hidden_size={self.hidden_size}, output_size={self.output_size}, "
                     f"num_layers={self.num_layers}, dropout_rate={self.dropout_rate}, "
                     f"activation={self.activation}, bidirectional={self.bidirectional}, "
                     f"feature_dim={self.feature_dim}")

        logger.info(f"Initializing DMHaloMatterRNN with input_size={self.input_size}, "
                     f"hidden_size={self.hidden_size}, output_size={self.output_size}, "
                     f"num_layers={self.num_layers}, dropout_rate={self.dropout_rate}, "
                     f"activation={self.activation}, bidirectional={self.bidirectional}, "
                     f"feature_dim={self.feature_dim}")
        
        # Initial batch normalization
        self.batch_norm = tcn.BatchNorm1d(self.feature_dim)
        #self.batch_norm = tcn.InstanceNorm1d(self.feature_dim) 
        
        # Feature extraction layers (optional pre-processing before RNN)
        self.feature_layer = tcn.Sequential(
            tcn.Linear(self.input_size, self.feature_dim),
            self.activation,
            tcn.Dropout(self.dropout_rate, inplace=True)
        )
        
        # Calculate output feature size (doubled if bidirectional)
        output_factor = 2 if self.bidirectional else 1
        intermediate_size = max(4, self.hidden_size // 2)  # Ensure at least size 4
        
        # Output projection layers
        self.output_layer = tcn.Sequential(
            tcn.Linear(self.hidden_size * output_factor, intermediate_size),
            self.activation,
            tcn.Dropout(self.dropout_rate, inplace=True),
            tcn.Linear(intermediate_size, self.output_size)
        )
        
        if self.use_layer_norm:
            self.norm_after_rnn = tcn.LayerNorm(self.hidden_size * output_factor)
    
    def __get_activation_layer(self, activation):
        """
        Get activation layer based on the specified activation function.
            
        Returns:
            torch.nn.Module: Activation layer.
        """
        activation_name = activation.lower()
    
        activation_map = {
            'relu': tcn.ReLU(inplace=True),
            'leaky_relu': tcn.LeakyReLU(inplace=True),
            'elu': tcn.ELU(),
            'tanh': tcn.Tanh(),
            'sigmoid': tcn.Sigmoid(),
            'gelu': tcn.GELU(),
            'selu': tcn.SELU(),
            'mish': tcn.Mish(),
            'hardshrink': tcn.Hardshrink(),
            'celu': tcn.CELU(),
            'swish': tcn.SiLU(inplace=True),  # PyTorch calls Swish as SiLU
            'glu': tcn.GLU(),
            'silu': tcn.SiLU(inplace=True),
        }
        
        if activation_name not in activation_map:
            logger.warning(f"Unknown activation function '{self.activation}', defaulting to GELU.")
        
        return activation_map.get(activation_name, tcn.GELU())
        
    def forward(self, x):
        """
        Forward pass (not implemented in base class).
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError("Forward is Not Implemented")
    
    '''    
    def process_input(self, x):
        """
        Processes input through batch normalization and feature layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size) or
                             (batch_size, seq_len, input_size).
        
        Returns:
            torch.Tensor: Processed tensor.
        """
        #logger.debug(f"Initial input shape: {x.shape}")
        assert x.dim() in [2, 3], f"Expected 2D or 3D input, got {x.dim()}D"

        normalize_fn = getattr(self, "normalize_input", None)
        if normalize_fn is not None:
            x = normalize_fn(x)
        
        if x.dim() == 2:
            # Input is (batch_size, input_size)
            x = self.feature_layers(x)  # → (batch_size, feature_dim)
            #logger.debug(f"After feature layers shape: {x.shape}")

            x = self.batch_norm(x)      # → (batch_size, feature_dim)
            #logger.debug(f"After batch_norm shape: {x.shape}")

            x = x.unsqueeze(1)          # → (batch_size, 1, feature_dim)
            #logger.debug(f"After unsqueeze shape: {x.shape}")

        else:
            # Input is (batch_size, seq_len, input_size)
            batch_size, seq_len, _ = x.shape

            #logger.debug(f"Input shape: {x.shape}")
                    
            x_reshaped = x.reshape(-1, self.input_size)       # (batch_size * seq_len, input_size)
            x_reshaped = self.feature_layers(x_reshaped)      # (batch_size * seq_len, feature_dim)
            x_reshaped = self.batch_norm(x_reshaped)          # (batch_size * seq_len, feature_dim)
            x = x_reshaped.reshape(batch_size, seq_len, self.feature_dim)
            
            
            x = x.contiguous().view(-1, self.input_size)          # (batch_size * seq_len, input_size)
            x = self.feature_layers(x)                            # (batch_size * seq_len, feature_dim)
            x = self.batch_norm(x)                                # (batch_size * seq_len, feature_dim)
            x = x.view(batch_size, seq_len, self.feature_dim)     # (batch_size, seq_len, feature_dim)
            #logger.debug(f"After feature + norm shape: {x.shape}")

        return x'''
        
    @tc.jit.ignore
    def process_input(self, x):
        """Optimized input processing"""
        # Handle 2D (single step) vs 3D (sequence) input
        is_sequence = x.dim() == 3
        
        if is_sequence:
            # Batch processing for sequences
            batch_size, seq_len, _ = x.shape
            x = x.reshape(-1, self.input_size)
            x = self.feature_layer(x)
            x = self.activation(x)
            
            # Apply batch norm efficiently
            x = self.batch_norm(x)
            
            # Return to sequence form
            return x.reshape(batch_size, seq_len, self.feature_dim)
        else:
            # Process single step
            x = self.feature_layer(x)
            x = self.activation(x)
            x = self.batch_norm(x)
            return x.unsqueeze(1)  # Add sequence dimension
    

    '''
    def process_input(self, x):
        """Simplified input processing"""
        if x.dim() == 2:
            # Single step: just transform features and add sequence dim
            x = self.feature_layers(x)
            x = x.unsqueeze(1)  # Add sequence dimension
        else:
            # Already has sequence: transform features
            batch_size, seq_len, _ = x.shape
            # More efficient reshape
            x = x.reshape(-1, self.input_size)
            x = self.feature_layers(x)
            x = x.reshape(batch_size, seq_len, self.feature_dim)
        
        # Apply batch norm only if needed (can be turned off for speed)
        if hasattr(self, 'batch_norm') and self.batch_norm is not None:
            if x.dim() == 3:
                # For 3D tensor, we need to reshape for batch norm
                orig_shape = x.shape
                x = x.reshape(-1, self.feature_dim)
                x = self.batch_norm(x)
                x = x.reshape(orig_shape)
            else:
                x = self.batch_norm(x)
        
        return x
    '''
    
    def apply_weight_init(self):
        """Apply optimized weight initialization."""
        for m in self.modules():
            if isinstance(m, tcn.Linear):
                #tcn.init.xavier_uniform_(m.weight)
                tcn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    tcn.init.zeros_(m.bias)
            
            elif isinstance(m, (tcn.RNN, tcn.LSTM, tcn.GRU)):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        tcn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        tcn.init.orthogonal_(param)
                    elif 'bias' in name:
                        tcn.init.zeros_(param)

    def clip_gradients(self, clip_value):
        tcn.utils.clip_grad_norm_(self.parameters(), clip_value)
        #logger.debug(f"Clipped gradients with clip value: {clip_value}")
    
    def summary(self, input_shape=(32, 10, None)):
        shape = list(input_shape)
        shape[2] = self.input_size
        return summary(self.to(self.device), input_size=tuple(shape), verbose=1)


class RNNDMHaloMapper(DMHaloMatterRNN):
    """
    Simple RNN model for dark matter halo mapping.
    
    Args:
        input_size (int): Input size.
        hidden_size (int): Hidden size.
        output_size (int): Output size.
        num_layers (int): Number of layers.
        dropout_rate (float): Dropout rate.
        activation (str): Activation function.
        bidirectional (bool): Whether to use bidirectional RNN.
        feature_dim (int): Dimension of feature space before RNN.
    
    Attrs:
        rnn (tcn.RNN): Recurrent neural network layer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Map activation name to RNN's nonlinearity parameter
        nonlinearity = 'tanh'  # RNN default
        if self.activation.lower() in ['relu', 'leaky_relu', 'elu', 'gelu', 'selu']:
            nonlinearity = 'relu'  # RNN only supports 'tanh' or 'relu'
        
        self.rnn = tcn.RNN(
                        self.feature_dim,  # Input is now feature_dim
                        self.hidden_size,
                        self.num_layers,
                        batch_first=True,
                        nonlinearity=nonlinearity,
                        dropout=self.dropout_rate if self.num_layers > 1 else 0,
                        bidirectional=self.bidirectional
                    )

        self.apply_weight_init()
        
        print(f"Initializing RNN with feature_dim={self.feature_dim}, "
                    f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
                    f"dropout_rate={self.dropout_rate}, nonlinearity={nonlinearity}, "
                    f"bidirectional={self.bidirectional}")
                
        logger.info(f"Initializing RNN with feature_dim={self.feature_dim}, "
                    f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
                    f"dropout_rate={self.dropout_rate}, nonlinearity={nonlinearity}, "
                    f"bidirectional={self.bidirectional}")
        
    def forward(self, x):
        """
        Forward pass through the RNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size) or
                             (batch_size, seq_len, input_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Process input through batch normalization and feature layers
        x = self.process_input(x)
        
        # Pass through RNN
        features, hidden = self.rnn(x)
        #logger.debug(f"After RNN shape: {features.shape}")
        
        # Get last time step (or concatenated if bidirectional)
        features = features[:, -1, :]
        #logger.debug(f"After slicing shape: {features.shape}")
        
        if self.use_layer_norm:
            features = self.norm_after_rnn(features)
            #logger.debug(f"After layer norm shape: {features.shape}")
        
        features = F.dropout(features, p=self.dropout_rate, training=self.training)
        #logger.debug(f"After dropout shape: {features.shape}")
        
        # Project to output space
        output = self.output_layer(features)
        return (output, hidden) if self.return_hidden else output

class LSTMDMHaloMapper(DMHaloMatterRNN):
    """
    LSTM model implementation for dark matter halo mapping.
    
    Args:
        input_size (int): Input size.
        hidden_size (int): Hidden size.
        output_size (int): Output size.
        num_layers (int): Number of layers.
        dropout_rate (float): Dropout rate.
        activation (str): Activation function for the output layers.
        bidirectional (bool): Whether to use bidirectional LSTM.
        feature_dim (int): Dimension of feature space before LSTM.
    
    Attrs:
        lstm (tcn.LSTM): Long Short-Term Memory (LSTM) network layer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = tcn.LSTM(
                        self.feature_dim,  # Input is now feature_dim
                        self.hidden_size,
                        self.num_layers,
                        batch_first=True,
                        dropout=self.dropout_rate if self.num_layers > 1 else 0,
                        bidirectional=self.bidirectional
                    )
        
        self.apply_weight_init()
        
        print(f"Initializing LSTM with feature_dim={self.feature_dim}, "
                    f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
                    f"dropout_rate={self.dropout_rate}, bidirectional={self.bidirectional}")
        
        logger.info(f"Initializing LSTM with feature_dim={self.feature_dim}, "
                    f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
                    f"dropout_rate={self.dropout_rate}, bidirectional={self.bidirectional}")
        
    def forward(self, x):
        """
        Forward pass through the LSTM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size) or
                             (batch_size, seq_len, input_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Process input through batch normalization and feature layers
        x = self.process_input(x)
        
        # Pass through LSTM
        features, (h_n, c_n) = self.lstm(x)
        #logger.debug(f"After LSTM shape: {features.shape}")
        
        # Get last time step (or concatenated if bidirectional)
        features = features[:, -1, :]
        #logger.debug(f"After slicing shape: {features.shape}")
        
        if self.use_layer_norm:
            features = self.norm_after_rnn(features)
            #logger.debug(f"After layer norm shape: {features.shape}")
        
        features = F.dropout(features, p=self.dropout_rate, training=self.training)
        #logger.debug(f"After dropout shape: {features.shape}")
        
        # Project to output space
        output = self.output_layer(features)
        return (output, (h_n, c_n)) if self.return_hidden else output
    
class GRUDMHaloMapper(DMHaloMatterRNN):
    """
    GRU model implementation for dark matter halo mapping.
    
    Args:
        input_size (int): Input size.
        hidden_size (int): Hidden size.
        output_size (int): Output size.
        num_layers (int): Number of layers.
        dropout_rate (float): Dropout rate.
        activation (str): Activation function for the output layers.
        bidirectional (bool): Whether to use bidirectional GRU.
        feature_dim (int): Dimension of feature space before GRU.
    
    Attrs:
        gru (tcn.GRU): Gated Recurrent Unit (GRU) network layer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gru = tcn.GRU(
                        self.feature_dim,  # Input is now feature_dim
                        self.hidden_size,
                        self.num_layers,
                        batch_first=True,
                        dropout=self.dropout_rate if self.num_layers > 1 else 0,
                        bidirectional=self.bidirectional
                    )
        
        self.apply_weight_init()

        # Log GRU initialization
        print(f"Initializing GRU with feature_dim={self.feature_dim}, "
                    f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
                    f"dropout_rate={self.dropout_rate}, bidirectional={self.bidirectional}")
        logger.info(f"Initializing GRU with feature_dim={self.feature_dim}, "
                    f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
                    f"dropout_rate={self.dropout_rate}, bidirectional={self.bidirectional}")

    def forward(self, x):
        """
        Forward pass through the GRU.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size) or
                             (batch_size, seq_len, input_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Process input through batch normalization and feature layers
        x = self.process_input(x)
        
        # Pass through GRU
        features, h_n = self.gru(x)
        #logger.debug(f"After GRU shape: {features.shape}")
        
        # Get last time step (or concatenated if bidirectional)
        # More efficient indexing for last time step
        if self.bidirectional:
            # For bidirectional, we need to get both directions' final states
            features = features[:, -1, :]
        else:
            # For unidirectional, just get the last time step
            features = features[:, -1, :]        
        #logger.debug(f"After slicing shape: {features.shape}")
        
        if self.use_layer_norm:
            features = self.norm_after_rnn(features)
            #logger.debug(f"After layer norm shape: {features.shape}")
        
        # Use functional dropout during training with inplace=True
        if self.training:
            features = F.dropout(features, p=self.dropout_rate, training=True, inplace=True)
        #logger.debug(f"After dropout shape: {features.shape}")
        
        # Project to output space
        output = self.output_layer(features)
        return (output, h_n) if self.return_hidden else output