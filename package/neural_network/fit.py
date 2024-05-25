from genericpath import isfile
from tabnanny import verbose
from typing import List, Tuple, Union
from matplotlib.axes import Axes
import torch
from torch import dropout, nn
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from .neural_network import ClassNetwork
from .util import get_dimension
from package.directories import weights_dir, eigenvalues_dir
from package.effective_dimension.effective_dimension import EDType, EffectiveDimensionApprox
from package.fim.fim import compute_fims_nn
from package.plot_utils.constant import DPI, SINGLE_PLOT_FIGSIZE


class FitOption:
    name: str

    input_dim: int
    layer_sizes: List[int]
    output_dim: int

    save_dirname: str

    dropout_rate: float

    def __init__(self, name: str, input_dim: int, layer_sizes: List[int], output_dim: int, save_dirname: str, dropout_rate: float=0.) -> None:
        self.name = name
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim
        self.save_dirname = save_dirname
        self.dropout_rate = dropout_rate


def calculate_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def fit_nn(
    input_dim: int,
    layer_sizes: List[int],
    output_dim: int,

    train_loader: DataLoader,
    test_loader: DataLoader,
    ed_loader: DataLoader,

    dropout_rate: float=0,
    num_thetas: int=100,
    eps: float=1e-3,
    eval_n: int=100000,
    gamma: float=1.0,
    chunk_size: int=100,

    skip_train: bool=False,

    num_epochs: int=1000,
    save_steps: int=10,

    dir_name: str='temp',
    num_runs: int=3,
    seeds: List[int]=[0, 24, 43],

    plot_eff_dims: bool=True,
    ax: Axes=None,
    label: str='',
    show: bool=True,
) -> Tuple[float, float, List[float]]:
    assert num_runs == len(seeds)

    save_epochs = list(range(0, num_epochs, num_epochs // save_steps))
    if save_epochs[-1] != num_epochs-1:
        save_epochs[-1] = num_epochs-1

    parent_dir = os.path.join(weights_dir, dir_name)
    if not os.path.isdir(parent_dir):
        os.mkdir(parent_dir)

    avg_gen_loss = 0
    avg_train_loss = 0
    if not skip_train:
        for run in range(num_runs):
            np.random.seed(seeds[run])
            torch.manual_seed(seeds[run])
            random.seed(seeds[run])

            run_dir = os.path.join(parent_dir, str(run))
            if not os.path.isdir(run_dir):
                os.mkdir(run_dir)

            network = ClassNetwork(input_dim, layer_sizes, output_dim, dropout_rate=dropout_rate)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

            # Training
            network.train()
            losses = []
            with tqdm(range(num_epochs), desc=f"Run: {run}. Epoch Progress", unit="epoch") as pbar:
                for epoch in pbar:
                    epoch_loss = 0
                    for i, (features, labels) in enumerate(train_loader):
                        features.requires_grad = True
                        outputs = network(features)

                        loss = criterion(outputs, labels)
                        epoch_loss += loss.item()

                        # Backward pass and optimization
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    avg_loss = epoch_loss / len(train_loader)

                    losses.append(avg_loss)

                    if epoch % 5 == 0 or epoch+1 == num_epochs:
                        pbar.set_description(
                                f"Run: {run}, Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
                        
                    if epoch in save_epochs:
                        network.eval()
                        model_path = os.path.join(
                            run_dir, f'epoch_{epoch + 1}.pth')
                        if os.path.isfile(model_path):
                            os.remove(model_path)
                        torch.save(network.state_dict(), model_path)
                        network.train()
            
            avg_train_loss += losses[-1]

            # Evaluation
            network.eval()
            test_loss = 0
            for i, (features, labels) in enumerate(test_loader):
                outputs = network(features)

                loss = criterion(outputs, labels)
                test_loss += loss.item()

            avg_loss = test_loss / len(test_loader)

            print(f'Generalization Loss: {abs(losses[-1] - avg_loss):.4f}')
            avg_gen_loss += abs(losses[-1] - avg_loss)

            del network
        
        avg_train_loss /= num_runs
        avg_gen_loss /= num_runs
        print('Average Genralization Error:', avg_gen_loss)

    eff_dims = []
    with tqdm(save_epochs, desc=f"Computing EDs", unit="epoch") as pbar:
        for epoch in pbar:
            for run in range(num_runs):
                network = ClassNetwork(input_dim, layer_sizes, output_dim, dropout_rate=dropout_rate)

                run_dir = os.path.join(parent_dir, str(run))
                model_path = os.path.join(
                        run_dir, f'epoch_{epoch + 1}.pth')
                
                network.load_state_dict(torch.load(model_path))
                network.eval()

                compute_fims_nn(
                    input_dim,
                    layer_sizes,
                    output_dim,

                    num_thetas=num_thetas,
                    save_dir=os.path.join(eigenvalues_dir, 'temp'),
                    filename=f'temp_{run}',

                    data_loader=ed_loader,

                    network=network,
                    theta_min=-eps,
                    theta_max=eps,

                    verbose=False
                )

                del network
            
            dim = get_dimension(input_dim, layer_sizes, output_dim)
            
            file_paths = [os.path.join(eigenvalues_dir, 'temp', f'temp_{run}_{dim}.h5') for run in range(num_runs)]
            ef = EffectiveDimensionApprox(file_paths, file_paths)
            eff_dims.append(
                ef.compute(eval_n, EDType.LOCAL, gamma=gamma, eps=eps, chunk_size=chunk_size, verbose=False)[0] / dim
            )

            pbar.update(1)

    if plot_eff_dims:
        plot = plt if ax is None else ax

        plot.plot(save_epochs, eff_dims, label=label) if label else plot.plot(save_epochs, eff_dims)
        
        if show:
            plot.show()

    return avg_gen_loss, avg_train_loss, eff_dims