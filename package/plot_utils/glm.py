from enum import Enum, auto
import math
import os
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pandas import DataFrame
import patsy
from tqdm import tqdm
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

from .constant import *
from package.glm.glm import deviance_loss
from package.fim.fim import compute_fims_glm
from package.effective_dimension.effective_dimension import EffectiveDimensionApprox, EDType
from package.directories import eigenvalues_dir


class ErrorType(Enum):
    TRAIN = auto()
    TEST = auto()
    GEN_ERR = auto()


def plot_train_and_test(
        x_train: NDArray,
        x_test: NDArray,

        y_train: NDArray,
        y_test: NDArray,

        betas: List[NDArray],

        step_size: int = 1,

        show: bool = True,
        ylim: Tuple[float, float] = None,

        ax: Axes = None,
):
    if ax is None:
        plt.figure(figsize=SINGLE_PLOT_FIGSIZE, dpi=DPI)

    train_losses = []
    test_losses = []
    steps = []
    for i in range(0, len(betas), step_size):
        train_losses.append(
            deviance_loss(betas[i], x_train, y_train)
        )
        test_losses.append(
            deviance_loss(betas[i], x_test, y_test)
        )
        steps.append(i)

    plot = ax if ax else plt
    plot.plot(steps, train_losses, label='train loss')
    plot.plot(steps, test_losses, label='test loss')
    plot.ylabel('Mean (scaled) Deviance Loss')
    plot.xlabel('$n$')
    plot.legend()

    if ylim:
        plot.ylim(ylim)

    if show:
        plt.show()


def plot_loss_and_ed(
    x_train: NDArray,
    x_test: NDArray,

    y_train: NDArray,
    y_test: NDArray,

    betas: List[NDArray],

    x_eval: NDArray,
    eps: float,

    n: int,

    save_dir: str,

    global_eff_dim: float = None,

    err_type: ErrorType = ErrorType.TRAIN,

    start_step: int = 0,
    num_betas: int = 1000,
    num_steps: int = 10,

    batch_size: int = 3000,

    title: str = 'Normalized effective Dimension and Mean (scaled) Deviance',

    show: bool = True,
    ax: Axes = None,
) -> Tuple[List[int], List[float]]:
    if ax is None:
        plt.figure(figsize=SINGLE_PLOT_FIGSIZE, dpi=DPI)

    if err_type == ErrorType.TRAIN:
        input_dim = x_train.shape[1]
    else:
        input_dim = x_test.shape[1]

    losses = []
    eff_dims = []
    steps = []

    step_size = max(1, math.ceil((len(betas) - start_step) / num_steps))

    progress_bar = tqdm(total=np.ceil(len(betas) / step_size),
                        desc=f'Computing EDs')
    for i in range(start_step, len(betas), step_size):
        if err_type == ErrorType.TRAIN:
            losses.append(
                deviance_loss(betas[i], x_train, y_train, batch_size)
            )
        elif err_type == ErrorType.TEST:
            losses.append(
                deviance_loss(betas[i], x_test, y_test, batch_size)
            )
        elif err_type == ErrorType.GEN_ERR:
            losses.append(
                abs(
                    deviance_loss(betas[i], x_train, y_train, batch_size) -
                    deviance_loss(betas[i], x_test, y_test, batch_size)
                )
            )
        else:
            raise ValueError('Invalid err_type')

        compute_fims_glm(
            input_dim,
            num_betas,
            features=x_eval,
            save_dir=save_dir,
            filename='glm_temp',

            beta_center=betas[i],
            beta_min=-eps,
            beta_max=eps,

            verbose=False
        )

        path = os.path.join(save_dir, f'glm_temp_{input_dim}.h5')

        ef = EffectiveDimensionApprox(
            path, path
        )

        eff_dim = ef.compute(n, EDType.LOCAL, gamma=1, eps=eps, verbose=False)

        eff_dims.append(eff_dim / input_dim)

        steps.append(i)
        progress_bar.update(1)

    progress_bar.close()

    plot = ax if ax else plt

    if ax is None:
        fig, ax1 = plot.subplots()
    else:
        fig = None
        ax1 = ax

    ax1.plot(steps, eff_dims, 'b-o',
             label='Local', markersize=4)
    if global_eff_dim is not None:
        ax1.plot(steps, np.ones_like(steps)*global_eff_dim,
                 label='Global', markersize=4)
        ax1.legend()
    ax1.set_ylabel('Normalized Effective Dimension', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlabel('Iteration')

    if global_eff_dim is None:
        ax2 = ax1.twinx()
        ax2.plot(steps, losses, 'r-o',
                 label='Mean (scaled) Deviance' if err_type != ErrorType.GEN_ERR else 'Generalization Error', markersize=4)
        ax2.set_ylabel('Mean (scaled) Deviance' if err_type !=
                       ErrorType.GEN_ERR else 'Generalization Error', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ned_margin = (max(eff_dims) - min(eff_dims)) * 0.1
        msd_margin = (max(losses) - min(losses)) * 0.1

        ax1.set_ylim(min(eff_dims) - ned_margin, max(eff_dims) + ned_margin)
        ax2.set_ylim(min(losses) - msd_margin, max(losses) + msd_margin)

    if ax is None:
        plot.title(title)
        fig.tight_layout()
    else:
        plot.set_title(title)

    if show:
        plt.show()

    return steps, eff_dims


def plot_steps_ed(
    results: List[GLMResultsWrapper],
    eps: float,
    num_betas: int,

    save_dirname: str,

    eval_df: DataFrame,
    n: int,

    plot_title: str,

    show: bool = True,
    ax: Axes = None,
    xlabel: str = 'Steps',
):
    save_dir = os.path.join(eigenvalues_dir, 'glm', save_dirname)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if ax is not None:
        plt.figure(figsize=SINGLE_PLOT_FIGSIZE)

    eff_dims = []
    for result in results:
        input_dim = len(result.params)

        _, x_eval = patsy.dmatrices(
            result.model.formula, eval_df, return_type='matrix')

        compute_fims_glm(
            input_dim,
            num_betas,
            beta_min=-eps,
            beta_max=eps,
            beta_center=result.params.values,
            features=x_eval,
            save_dir=save_dir,
            filename='temp'
        )

        path = os.path.join(save_dir, f'temp_{input_dim}.h5')

        ef = EffectiveDimensionApprox(
            path, path
        )

        eff_dim = ef.compute(n, EDType.LOCAL, gamma=1, eps=eps, verbose=False)

        eff_dims.append(eff_dim / input_dim)

    plt.plot(range(len(results)), eff_dims,
             '-o') if ax is None else ax.plot(range(len(results)), eff_dims, '-o')
    plt.ylabel('Normalized Effective Dimension') if ax is None else ax.set_ylabel(
        'Normalized Effective Dimension')
    plt.xlabel(xlabel) if ax is None else ax.set_xlabel(xlabel)
    plt.xticks(range(len(results)))
    plt.title(plot_title) if ax is None else ax.set_title(plot_title)

    if show:
        plt.show()
