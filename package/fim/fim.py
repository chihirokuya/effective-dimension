import os
import random
from typing import Callable, List, Tuple
import numpy as np
from scipy.special import factorial
from numpy.typing import NDArray
import h5py
import concurrent.futures
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC
import logging

from package.neural_network.neural_network import ClassNetwork
from package.neural_network.util import get_dimension


logging.basicConfig(level=logging.INFO)


def _compute_fim_eigenvalues_nn(
        input_dim: int,
        layer_sizes: list[int],
        output_dim: int,
        data_loader: DataLoader,
        theta: NDArray,
        dropout_rate: float = 0,
) -> NDArray:
    """
    Computes the eigenvalues of the Fisher Information Matrix (FIM) for a neural network.

    Args:
        input_dim (int): The input dimension.
        layer_sizes (list[int]): A list of integers representing the sizes of the hidden layers.
        output_dim (int): The output dimension.
        data_loader (DataLoader): The DataLoader containing the dataset.
        theta (NDArray): The parameter values for the neural network.
        dropout_rate (float, optional): The dropout rate for the neural network. Default is 0.

    Returns:
        NDArray: An array of eigenvalues of the Fisher Information Matrix.
    """
    first_batch = next(iter(data_loader))
    if isinstance(first_batch, tuple):
        device = first_batch[0].device
    else:
        device = first_batch.device

    convnet = ClassNetwork(input_dim, layer_sizes,
                           output_dim, dropout_rate)

    convnet.set_weights(theta)

    convnet = convnet.to(torch.device(device))

    fim = FIM(model=convnet,
              loader=data_loader,
              representation=PMatKFAC,
              device=device,
              n_output=output_dim
              )

    return fim.get_diag().cpu().numpy()


def _compute_fim_eigenvalues_glm_batched(
        features: NDArray,
        beta: NDArray,

        response: NDArray = None,
        cap: int = None,
        batch_size: int = 0,
) -> NDArray:
    """
    Computes the eigenvalues of the Fisher Information Matrix (FIM) for the Poisson Generalized Linear Model (GLM) in batches.

    Args:
        features (NDArray): The input features for the GLM.
        beta (NDArray): The parameter values for the GLM.
        response (NDArray, optional): The response variable for the GLM. Default is None. Required if cap is not None.
        cap (int, optional): The maximum value to cap the response variable at. Default is None.
        batch_size (int, optional): The number of samples per batch. Default is 0, which means no batching.

    Returns:
        NDArray: An array of eigenvalues of the Fisher Information Matrix.
    """
    num_samples = features.shape[0]
    FIM: NDArray = None
    FIM_batch: NDArray = None

    if not batch_size:
        batch_size = num_samples

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        features_batch = features[start:end]

        exp_eta = np.exp(features_batch @ beta)

        multiplier = exp_eta
        if isinstance(cap, int):
            response_batch = response[start:end]
            capped_indices = np.where(response_batch >= cap)[0]

            if capped_indices.size > 0:
                exp_eta_capped = exp_eta[capped_indices]

                # Compute G for the batch
                prob_cap = exp_eta_capped ** cap / \
                    factorial(cap) * np.exp(-exp_eta_capped)
                k = np.arange(cap)
                exp_eta_expanded = exp_eta_capped[:, np.newaxis]
                k_power_exp_eta = np.power(exp_eta_expanded, k)

                F_cap = np.sum(
                    k_power_exp_eta / factorial(k) * np.exp(-exp_eta_expanded),
                    axis=1)

                G = prob_cap / (1 - F_cap)

                multiplier[capped_indices] = - cap * (
                    (cap - exp_eta_capped) * G - cap * G**2
                )

        # Calculate FIM for the batch
        FIM_batch = np.einsum('ij,ik,i->ijk', features_batch,
                              features_batch, multiplier)

        if FIM is None:
            FIM = FIM_batch
        elif FIM.shape == FIM_batch.shape:
            FIM += FIM_batch

    # Take weighted mean
    if num_samples % batch_size:
        weight = FIM_batch.shape[0] / num_samples
        FIM = (1 - weight) * np.mean(FIM, axis=0) / (num_samples // batch_size) \
            + weight * np.mean(FIM_batch, axis=0)
    else:
        FIM = np.mean(FIM, axis=0) / (num_samples // batch_size)

    eigenvalues = np.linalg.eigvalsh(FIM)

    return eigenvalues


def _generate_thetas(
        num_samples: int,
        input_dim: int,

        # Neural Network
        layer_sizes: List[int] = None,
        output_dim: int = 0,

        theta_max: float = 1,
        theta_min: float = -1,
) -> NDArray:
    """
    Generates a set of uniformly and randomly theta values.
    If abs(theta_min) < 1, then theta values are drawn from a sphere.

    Args:
        num_samples (int): The number of samples to generate.
        input_dim (int): The input dimension.
        layer_sizes (List[int], optional): The list of layer sizes for a neural network. If None, generates for simple input dimension.
        output_dim (int): The output dimensionality of the neural network. Default is 0.
        theta_max (float): The maximum value for theta. Default is 1.
        theta_min (float): The minimum value for theta. Default is -1.

    Returns:
        NDArray: An array of theta values with shape (num_samples, dimension).
    """
    assert theta_min < theta_max

    if isinstance(layer_sizes, list):
        dimension = get_dimension(input_dim, layer_sizes, output_dim)
    else:
        dimension = input_dim

    if abs(theta_min) < 1:
        points = np.random.normal(size=(num_samples, dimension))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        points *= theta_min

        return points
    else:
        return np.random.uniform(theta_min, theta_max, size=(num_samples, dimension))


def _generate_betas_w_con(
        num_betas: int,
        input_dim: int,
        beta_min: float,
        beta_max: float,
        features: NDArray,

        con_min: float = 0,
        con_max: float = np.inf,
        batch_size: int = 0,
) -> NDArray:
    """
    Generates beta values with constraints on the exponential of the dot product of beta and features.

    Args:
        num_betas (int): The number of beta values to generate.
        input_dim (int): The input dimension.
        beta_min (float): The minimum value for beta.
        beta_max (float): The maximum value for beta.
        features (NDArray): The input features for the GLM.
        con_min (float, optional): The minimum constraint on exp(<beta, X>). Default is 0.
        con_max (float, optional): The maximum constraint on exp(<beta, X>). Default is np.inf.
        batch_size (int, optional): The number of samples per batch. Default is 0, which means no batching.

    Returns:
        NDArray: An array of beta values that satisfy the constraints.
    """
    if not batch_size:
        batch_size = features.shape[0]

    unique_betas = set()
    while len(unique_betas) < num_betas:
        betas = _generate_thetas(num_betas - len(unique_betas), input_dim,
                                 theta_min=beta_min, theta_max=beta_max)

        # Initially assume all betas are valid
        valid_betas_mask = np.ones(betas.shape[0], dtype=bool)

        for start in range(0, num_betas, batch_size):
            end = start + batch_size
            features_batch = features[start:end]
            exp_eta = np.exp(betas @ features_batch.T)

            # Update valid betas based on this batch
            batch_valid = np.all((exp_eta > con_min) &
                                 (exp_eta < con_max), axis=1)
            valid_betas_mask &= batch_valid

        unique_betas.update(map(tuple, betas[valid_betas_mask]))

        if len(unique_betas) >= num_betas:
            unique_betas = set(list(unique_betas)[:num_betas])
            break

    return np.array(list(unique_betas))


def _get_eigenvalues(
        thetas: NDArray,
        compute_fim_eigenvalues: Callable,
        tasks: List[dict],
        file_path: str = '',
        store: bool = True,

        verbose: bool = True,
) -> list[NDArray]:
    """
    Computes the eigenvalues of the Fisher Information Matrix (FIM) for given parameters and tasks,
    optionally storing the results in an HDF5 file.

    Args:
        thetas (NDArray): The parameter values for which to compute the eigenvalues.
        compute_fim_eigenvalues (Callable): The function to compute the FIM eigenvalues.
        tasks (List[dict]): A list of dictionaries, where each dictionary contains the parameters
                            for a single call to compute_fim_eigenvalues.
        file_path (str, optional): The path to the file where results will be stored. Required if store is True.
        store (bool, optional): Whether to store the results in an HDF5 file. Default is True.
        verbose (bool, optional): Whether to display a progress bar. Default is True.

    Returns:
        list[NDArray]: A list of arrays, where each array contains the eigenvalues of the FIM for a set of parameters.
    """
    assert not store or file_path != ''
    with concurrent.futures.ThreadPoolExecutor() as executor:
        if verbose:
            results = list(tqdm(executor.map(lambda p: compute_fim_eigenvalues(**p), tasks),
                                total=len(tasks),
                                desc="Computing Eigenvalues"))
        else:
            results = executor.map(
                lambda p: compute_fim_eigenvalues(**p), tasks)

    if store:
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('thetas', data=thetas,
                             compression="gzip", chunks=True)
            eigenvalues_group = f.create_group('eigenvalues')

            for i, eigenvalues in enumerate(results):
                eigenvalues_group.create_dataset(
                    f'index_{i}', data=eigenvalues, compression="gzip")

    return np.array(results)


def compute_fims_nn(
    input_dim: int,
    layer_sizes: list[int],
    output_dim: int,

    num_thetas: int,

    data_loader: DataLoader,

    network: ClassNetwork = None,
    dropout_rate: float = 0,

    theta_min: float = -1,
    theta_max: float = 1,
    theta_center: NDArray = None,

    num_samples: int = 0,

    save_dir: str = "",
    filename: str = "",
    store: bool = True,

    verbose: bool = True,

    seed: int=42,
) -> Tuple[NDArray, List[NDArray]]:
    """
    Computes the Fisher Information Matrices (FIMs) for a neural network over a range of parameter values.

    Args:
        input_dim (int): The input dimension.
        layer_sizes (list[int]): A list of integers representing the sizes of the hidden layers.
        output_dim (int): The output dimension.
        num_thetas (int): The number of parameter sets (thetas) to generate.
        data_loader (DataLoader): The DataLoader containing the dataset.
        network (ClassNetwork, optional): The neural network class to use. If None, a new network will be created. Default is None.
        dropout_rate (float, optional): The dropout rate for the neural network. Default is 0.
        theta_min (float, optional): The minimum value for theta. Default is -1.
        theta_max (float, optional): The maximum value for theta. Default is 1.
        theta_center (NDArray, optional): The central value for theta around which to generate samples. Default is None.
        num_samples (int, optional): The number of samples to use. Default is 0.
        save_dir (str, optional): The directory to save the results. Default is "".
        filename (str, optional): The filename to save the results. Default is "".
        store (bool, optional): Whether to store the results in a file. Default is True.
        verbose (bool, optional): Whether to display a progress bar. Default is True.
        seed (int, optional): The random seed for reproducibility. Default is 42.

    Returns:
        Tuple[NDArray, List[NDArray]]: A tuple containing the generated thetas and a list of FIM eigenvalues.
    """
    assert not store or (filename and save_dir), "file_path and save_dir is required if store=True"
    assert data_loader is not None or num_samples, "num_samples is required if data_loader=None"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    thetas = _generate_thetas(
        num_thetas,
        input_dim,
        layer_sizes=layer_sizes,
        output_dim=output_dim,
        theta_min=theta_min,
        theta_max=theta_max
    )

    if network:
        all_parameters = np.array([])
        for param in network.parameters():
            all_parameters = np.concatenate(
                (all_parameters, param.clone().detach().numpy().reshape(-1)))
        thetas += np.array(all_parameters)
    elif theta_center is not None:
        thetas += theta_center

    tasks = [
        {
            'input_dim': input_dim,
            'layer_sizes': layer_sizes,
            'output_dim': output_dim,
            'data_loader': data_loader,
            'theta': theta,
            'dropout_rate': dropout_rate,
        }
        for theta in thetas
    ]
    
    if store:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        file_path = os.path.join(save_dir, f'{filename}_{thetas.shape[1]}.h5')

    results = _get_eigenvalues(thetas,
                               _compute_fim_eigenvalues_nn,
                               tasks,
                               file_path=file_path,
                               store=store,
                               verbose=verbose)

    if verbose:
        print(np.mean(results.max(axis=1)), np.mean(results.min(axis=1)),
              np.mean(results.var(axis=1) / np.mean(results.max(axis=1))))

    return thetas, results


def compute_fims_glm(
    input_dim: int,

    num_betas: int,

    beta_min: float = -1,
    beta_max: float = 0,
    beta_center: NDArray = None,
    betas: NDArray = None,

    con_min: float = 0,
    con_max: float = np.inf,

    features: NDArray = None,
    num_samples: int = 0,

    save_dir: str="",
    filename: str = "",
    store: bool = True,

    response: NDArray = None,
    cap: int = None,

    batch_size: int = 0,

    verbose: bool = True,

    seed: int=42
) -> Tuple[NDArray, List[NDArray]]:
    """
    Computes the Fisher Information Matrices (FIMs) for the Poisson Generalized Linear Model (GLM) over a range of parameter values.

    Args:
        input_dim (int): The input dimension.
        num_betas (int): The number of beta values to generate.
        beta_min (float, optional): The minimum value for beta. Default is -1.
        beta_max (float, optional): The maximum value for beta. Default is 0.
        beta_center (NDArray, optional): The central value for beta around which to generate samples. Default is None.
        betas (NDArray, optional): Predefined beta values. If provided, generation of betas is skipped. Default is None.
        con_min (float, optional): The minimum constraint on exp(<beta, X>). Default is 0.
        con_max (float, optional): The maximum constraint on exp(<beta, X>). Default is np.inf.
        features (NDArray, optional): The input features for the GLM. Default is None.
        num_samples (int, optional): The number of samples to use for beta generation. Default is 0.
        save_dir (str, optional): The directory to save the results. Default is "".
        filename (str, optional): The filename to save the results. Default is "".
        store (bool, optional): Whether to store the results in a file. Default is True.
        response (NDArray, optional): The response variable for the GLM. Default is None.
        cap (int, optional): The maximum value to cap the response variable at. Default is None.
        batch_size (int, optional): The number of samples per batch for FIM computation. Default is 0.
        verbose (bool, optional): Whether to display a progress bar. Default is True.
        seed (int, optional): The random seed for reproducibility. Default is 42.

    Returns:
        Tuple[NDArray, List[NDArray]]: A tuple containing the generated beta values and a list of FIM eigenvalues.
    """
    assert not store or (filename and save_dir), "file_path and save_dir is required if store=True"
    assert features is not None or num_samples, "num_samples is required if features=None"
    assert not cap or response is not None, "response is required if cap!=None"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if features is None:
        features = np.random.normal(0, 1, (num_samples, input_dim))

    if betas is None:
        betas = _generate_betas_w_con(
            num_betas,
            input_dim,
            beta_min,
            beta_max,
            features,
            con_min,
            con_max,
            batch_size
        )

    if beta_center is not None:
        betas += beta_center

    tasks = [
        {
            'features': features,
            'beta': beta,
            'response': response,
            'cap': cap,
            'batch_size': batch_size,
        }
        for beta in betas
    ]

    if store:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        file_path = os.path.join(save_dir, f'{filename}_{betas.shape[1]}.h5')

    if cap is not None:
        print(f'proportion cap={cap}:', np.count_nonzero(
            response >= cap) / len(response))

    results = _get_eigenvalues(betas,
                               _compute_fim_eigenvalues_glm_batched,
                               tasks,
                               file_path=file_path,
                               store=store,
                               verbose=verbose)

    if verbose:
        print(np.mean(results.max(axis=1)), np.mean(results.min(axis=1)),
              np.mean(results.var(axis=1) / results.max(axis=1)))

    return betas, results
