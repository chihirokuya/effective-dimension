from typing import List


def get_dimension(input_dim: int, layer_sizes: List[int], output_dim: int) -> int:
    """
    Computes the total number of parameters within a neural network.

    Args:
        input_dim (int): The dimensionality of the input layer.
        layer_sizes (List[int]): A list of integers representing the sizes of the hidden layers.
        output_dim (int): The dimensionality of the output layer.

    Returns:
        int: The total number of parameters in the neural network.
    """
    dimension = input_dim * layer_sizes[0]
    dimension += layer_sizes[0]  # Bias

    for i in range(len(layer_sizes) - 1):
        dimension += layer_sizes[i] * layer_sizes[i+1]
        dimension += layer_sizes[i+1]  # Bias

    dimension += layer_sizes[-1] * output_dim
    dimension += output_dim  #

    return dimension
