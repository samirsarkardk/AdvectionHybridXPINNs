import numpy as np
import torch

class Config:
    def __init__(self):
        # Set random seed for reproducibility
        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        # Training parameters
        self.num_epochs = 1000
        self.learning_rate = 0.001

        # Model parameters
        self.input_dim = 2  
        self.num_hidden_layers = 3
        self.hidden_dim = 20
        self.output_dim = 1
        self.activation_function = torch.nn.Tanh()

        # Domain configuration
        self.x_min = 0.0
        self.x_max = 1.0
        self.t_min = 0.0
        self.t_max = 1.0
        self.delta = 0.05