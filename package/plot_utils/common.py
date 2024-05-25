from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Union, List, Tuple


from package.effective_dimension.effective_dimension import EffectiveDimensionApprox, ComputeMethod, EDType
from .constant import *


class CompareBy:
    DIMENSION = auto()
    DROPOUT_RATE = auto()
    LABEL = auto()


class AxisOption(Enum):
    PLOT = auto()
    SEMILOGX = auto()
    SEMILOGY = auto()
    LOGLOG = auto()
    SCATTER = auto()


class PlotOption:
    file_path: str
    label: str
    compute_method: ComputeMethod
    ed_type: EDType
    gamma: float
    eps: float

    normalized: bool
    dim: int
    dropout_rate: float

    cache_eigenvalues: bool
    chunk_size: int

    verbose: bool

    def __init__(self,
                 file_path: str,
                 label: str,
                 compute_method: ComputeMethod = ComputeMethod.EFFICIENT,
                 ed_type: EDType = EDType.GLOBAL,
                 gamma: float = 1,
                 eps: float = 0,

                 normalized: bool = False,
                 dim: int = 0,
                 dropout_rate: float = 0,

                 cache_eigenvalues: bool = True,
                 chunk_size: int = 100,

                 verbose: bool = True,
                 ):
        assert not normalized or dim

        self.file_path = file_path
        self.label = label
        self.compute_method = compute_method
        self.ed_type = ed_type
        self.gamma = gamma
        self.eps = eps

        self.normalized = normalized
        self.dim = dim
        self.dropout_rate = dropout_rate

        self.cache_eigenvalues = cache_eigenvalues
        self.chunk_size = chunk_size

        self.verbose = verbose


class PlotUtil:
    @staticmethod
    def plot(
            options: Union[PlotOption, List[PlotOption]],
            num_points: int = 1000,
            n_min: int = int(1e3),
            n_max: int = int(1e5),

            title: str = '',
            ylabel: str = 'Effective Dimension',

            axis_option: AxisOption = AxisOption.SEMILOGX,
            ylim: Tuple[float, float] = None,

            ax: Axes = None,
            show: bool = True,
    ):
        plot = plt if ax is None else ax

        if ax is None:
            plt.figure(figsize=SINGLE_PLOT_FIGSIZE, dpi=DPI)

        ns = np.linspace(n_min, n_max, num_points).astype(int)

        if isinstance(options, PlotOption):
            options = [options]

        for option in options:
            if option.ed_type == EDType.LOCAL:
                ef = EffectiveDimensionApprox(
                    option.file_path,
                    local_file_path=option.file_path,
                    cache_eigenvalues=option.cache_eigenvalues
                )
            else:
                ef = EffectiveDimensionApprox(
                    option.file_path, cache_eigenvalues=option.cache_eigenvalues)

            eds = ef.compute(
                n=ns,
                ed_type=option.ed_type,
                compute_method=option.compute_method,
                gamma=option.gamma,
                eps=option.eps,
                chunk_size=option.chunk_size,
                verbose=option.verbose
            )

            del ef

            if option.normalized:
                eds = np.array(eds)/option.dim

            if axis_option == AxisOption.SEMILOGX:
                plot.semilogx(ns, eds, label=option.label,
                              marker='o', markersize=2)
            elif axis_option == AxisOption.SEMILOGY:
                plot.semilogy(ns, eds, label=option.label,
                              marker='o', markersize=2)
            elif axis_option == AxisOption.PLOT:
                plot.plot(ns, eds, label=option.label,
                          marker='o', markersize=2)
            elif axis_option == AxisOption.LOGLOG:
                plot.loglog(ns, eds, label=option.label,
                            marker='o', markersize=2)
            elif axis_option == AxisOption.SCATTER:
                plot.scatter(ns, eds, label=option.label, s=2)
            else:
                raise ValueError('Invalid axis option')

        if ax is None:
            if title:
                plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel('Sample Size $n$')
            plt.legend()
            if ylim:
                plt.ylim(ylim)
        else:
            if title:
                ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Sample Size $n$')
            ax.legend()

            if ylim:
                ax.set_ylim(ylim)

        if show and ax is None:
            plt.show()

    @staticmethod
    def compare_with_fixed_n(
        options: Union[PlotOption, List[PlotOption]],
        n: int,

        compare_by: CompareBy,

        title: str = None,
        ylabel: str = 'Effective Dimension',
        label: str = None,
        legend_loc: str = None,
        losses: List[float] = None,
        loss_label: str='Mean (Scaled) Deviance',

        ax: Axes = None,
        show: bool = True,
    ):
        assert isinstance(n, int), "n must be an integer"

        if ax is None:
            plt.figure(figsize=SINGLE_PLOT_FIGSIZE, dpi=DPI)

        if isinstance(options, PlotOption):
            options = [options]

        xticks = []
        eff_dims = []

        for option in options:
            if option.ed_type == EDType.LOCAL:
                ef = EffectiveDimensionApprox(
                    option.file_path,
                    local_file_path=option.file_path,
                    cache_eigenvalues=option.cache_eigenvalues
                )
            else:
                ef = EffectiveDimensionApprox(
                    option.file_path, cache_eigenvalues=option.cache_eigenvalues)

            eds = ef.compute(
                n=n,
                ed_type=option.ed_type,
                compute_method=option.compute_method,
                gamma=option.gamma,
                eps=option.eps,
                chunk_size=option.chunk_size,
                verbose=option.verbose
            )

            del ef

            if option.normalized:
                eds = np.array(eds)/option.dim

            if compare_by == CompareBy.DIMENSION:
                xticks.append(option.dim)
            elif compare_by == CompareBy.DROPOUT_RATE:
                xticks.append(option.dropout_rate)
            elif compare_by == CompareBy.LABEL:
                xticks.append(option.label)
            else:
                raise ValueError("Invalid compare_by")

            eff_dims.append(eds[0])

        sorted_idx = np.argsort(xticks)
        xticks = np.array(xticks)[sorted_idx]
        eff_dims = np.array(eff_dims)[sorted_idx]

        if compare_by == CompareBy.DIMENSION:
            xlabel = '$d$'
        elif compare_by == CompareBy.DROPOUT_RATE:
            xlabel = 'Dropout Rate'
        elif compare_by == CompareBy.LABEL:
            xlabel = ''

        ax1: Axes
        marker: str
        if losses is not None:
            _, ax1 = plt.subplots() if ax is None else ax.subplots()
            marker = 'b-o'
        else:
            ax1 = ax
            marker = '-o'

        (
            plt.plot(range(len(xticks)), eff_dims, marker, markersize=4, label=label) if ax is None else
            ax1.plot(range(len(xticks)), eff_dims,
                    marker, markersize=4, label=label)
        )

        if label:
            if legend_loc:
                plt.legend(loc=legend_loc) if ax is None else ax1.legend(
                    loc=legend_loc)
            else:
                plt.legend() if ax is None else ax1.legend()

        plt.xlabel(xlabel) if ax is None else ax1.set_xlabel(xlabel)
        if title:
            plt.title(title) if ax is None else ax1.set_title(title)
        plt.xticks(range(len(xticks)), xticks) if ax is None else ax1.set_xticks(
            range(len(xticks)), xticks)
        
        if losses is not None:
            ax1.set_ylabel(ylabel, color='b')
            ax1.tick_params(axis='y', labelcolor='b')

            losses = np.array(losses)[sorted_idx]
            ax2 = ax1.twinx()
            ax2.plot(range(len(xticks)), losses, 'r-o', label=loss_label)
            ax2.set_ylabel(loss_label, color='r')
            ax2.tick_params(axis='y', labelcolor='r')
        else:
            plt.ylabel(ylabel) if ax is None else ax1.set_ylabel(ylabel)

        if show:
            plt.show()
