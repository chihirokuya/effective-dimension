from typing import List, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
from scipy.stats import poisson


def upper_bound(
    ns: Union[int, List[int]], c: float,
    eff_dims: Union[float, List[float]],
    M: float,
    B: float,

    gamma: float = 1,
    alpha: float = 1,

    use_log: bool = False,

    eps: float=0,
) -> NDArray[np.float32]:
    if isinstance(ns, int):
        ns = [ns]
    if isinstance(eff_dims, float):
        eff_dims = [eff_dims]

    assert len(ns) == len(eff_dims), "len(n) != len(eff_dims)"

    ns = np.array(ns)
    eff_dims = np.array(eff_dims)

    if use_log:
        log_second_term = eff_dims/2 * np.log(gamma * np.power(ns, 1/alpha) /
                                              (2 * np.pi * np.log2(np.power(ns, 1/alpha))))

        log_third_term = -16 * M**2 * np.pi * np.log2(ns) / (B**2 * gamma)

        if eps > 0:
            log_third_term *= eps ** 2

        return np.log(c) + log_second_term + log_third_term
    else:
        second_term = np.power(gamma * np.power(ns, 1/alpha) /
                               (2 * np.pi * np.log2(np.power(ns, 1/alpha))), eff_dims/2)

        third_term = np.exp(-16 * M**2 * np.pi * np.log2(ns) / (B**2 * gamma))

        if eps > 0:
            log_third_term *= np.exp(eps ** 2)

        return c * second_term * third_term


def glm_expected_loss(beta_a: float, beta_b: float):
    def e_y_logy(a: float, b: float):
        lam = np.exp(-0.9 * a - 0.8 * b)
        sum_poisson = sum(y * np.log(y) * poisson.pmf(y, lam) for y in range(1, 100))
        return sum_poisson

    def e_exp(beta: float):
        return (np.exp(beta*1.5) - np.exp(beta*0.5)) / beta

    def y_betax(beta_a, beta_b):
        def part1(a):
            return -a / 0.9 * np.exp(-0.9 * a) - 1 / 0.81 * np.exp(-0.9 * a)
        
        part1_val = part1(1.5) - part1(0.5)

        def part4(b):
            return -b / 0.8 * np.exp(-0.8 * b) - 1 / 0.64 * np.exp(-0.8 * b)
        
        part4_val = part4(1.5) - part4(0.5)

        result = beta_a * part1_val * e_exp(-0.8) + beta_b * e_exp(-0.9) * part4_val
        
        return result

    first_term = dblquad(e_y_logy, 0.5, 1.5, lambda x: 0.5, lambda x: 1.5)[0]
    second_term = - y_betax(beta_a, beta_b)
    third_term = - e_exp(-0.9) * e_exp(-0.8)
    fourth_term = e_exp(beta_a) * e_exp(beta_b)

    return 2 * (first_term + second_term + third_term + fourth_term)


class LowerBoundPlotOption:
    M: float
    gamma: float
    ns: Union[int, List[int]]

    def __init__(self, ns: Union[int, List[int]], M: float, gamma: float) -> None:
        self.ns = ns
        self.M = M
        self.gamma = gamma


class UpperBoundPlotOption:
    ns: Union[int, List[int]]
    eff_dims: Union[float, List[float]]
    M: float
    B: float
    c: float
    gamma: float
    alpha: float

    use_log: bool

    def __init__(self,
                 ns: Union[int, List[int]],
                 eff_dims: Union[float, List[float]],
                 M: float,
                 B: float,
                 c: float,
                 gamma: float,

                 alpha: float=1.,

                 use_log: bool = True,
                 ) -> None:
        self.ns = ns
        self.eff_dims = eff_dims
        self.M = M
        self.c = c
        self.B = B
        self.gamma = gamma
        self.alpha = alpha

        self.use_log = use_log


def plot_lowerbound(
    options: Union[LowerBoundPlotOption, List[LowerBoundPlotOption]],
    figsize: Tuple[int, int] = (10, 8)
):
    if isinstance(options, LowerBoundPlotOption):
        options = [options]

    num_rows = 2
    num_cols = 2
    max_plots_per_fig = num_rows * num_cols

    num_options = len(options)
    num_figures = (num_options + max_plots_per_fig - 1) // max_plots_per_fig
    
    option_index = 0

    for fig_num in range(num_figures):
        plt.figure(figsize=figsize)

        for subplot_index in range(1, max_plots_per_fig + 1):
            if option_index >= num_options:
                break

            option = options[option_index]

            plt.subplot(num_rows, num_cols, subplot_index)
            plt.semilogx(option.ns, 
                        4 * option.M * np.sqrt(2 * np.pi * np.log2(option.ns) / (option.gamma * option.ns)),
                     label=f'$\gamma$={option.gamma}, M={option.M}')
            plt.title(
                f'Lower Bound: $M={option.M}$, $\gamma={option.gamma}$')
            plt.xlabel('$n$')
            plt.ylabel('Lower Bound')
            plt.legend()

            option_index += 1

        plt.tight_layout()
        plt.show()
        plt.close()


def plot_bounds(
    options: Union[UpperBoundPlotOption, List[UpperBoundPlotOption]],

    figsize: Tuple[int, int] = (10, 8),
):
    if isinstance(options, UpperBoundPlotOption):
        options = [options]

    num_rows = 2
    num_cols = 2
    max_plots_per_fig = num_rows * num_cols

    num_options = len(options)
    num_figures = (num_options + max_plots_per_fig - 1) // max_plots_per_fig
    
    option_index = 0

    for _ in range(num_figures):
        plt.figure(figsize=figsize)

        for subplot_index in range(1, max_plots_per_fig + 1):
            if option_index >= num_options:
                break

            option = options[option_index]

            plt.subplot(num_rows, num_cols, subplot_index)
            plt.semilogx(option.ns, 
                     upper_bound(option.ns, option.c, option.eff_dims, option.M, option.B, option.gamma, option.alpha, use_log=option.use_log),
                     label=f'$\gamma$={option.gamma}, M={option.M}')
            title = '(log) Upper Bound' if option.use_log else 'Upper Bound'
            plt.title(f'{title}: $M={option.M}$, $\gamma={option.gamma}$')
            plt.xlabel('$n$')
            plt.ylabel(title)
            plt.legend()

            option_index += 1

        plt.tight_layout()
        plt.show()
        plt.close()
