import torch
from torch import nn
import numpy as np
from numpy.typing import NDArray

class ClassNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 layer_sizes: list[int],
                 output_dim: int,
                 dropout_rate: float = 0
                 ):
        super(ClassNetwork, self).__init__()

        layers = []
        current_dim = input_dim

        for size in layer_sizes:
            layers.append(nn.Linear(current_dim, size))
            layers.append(nn.LeakyReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = size

        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        if len(x.shape) > 1:
            x = x.view(x.size(0), -1)
        x = self.network(x)
        return x
    
    def extract_weights(self) -> NDArray:
        all_parameters = np.array([])
        for param in self.parameters():
            all_parameters = np.concatenate((all_parameters, param.clone().detach().numpy().reshape(-1)))
        return all_parameters

    def set_weights(self, theta):
        with torch.no_grad():
            offset = 0
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    # Calculate weight and bias sizes for this layer
                    next_dim = layer.out_features
                    weight_count = next_dim * layer.in_features
                    bias_count = next_dim

                    # Reshape weights and biases
                    layer_weights = theta[offset:offset +
                                          weight_count].reshape((next_dim, layer.in_features))
                    layer_biases = theta[offset +
                                         weight_count:offset + weight_count + bias_count]

                    # Set the parameters
                    layer.weight = nn.Parameter(
                        torch.from_numpy(layer_weights).float())
                    layer.bias = nn.Parameter(
                        torch.from_numpy(layer_biases).float())

                    # Update offset for next layer
                    offset += weight_count + bias_count


