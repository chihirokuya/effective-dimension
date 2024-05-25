from abc import ABC, abstractmethod
import os
from typing import List, Union
import h5py
import numpy as np
from numpy.typing import NDArray
from enum import Enum, auto

from tqdm import tqdm


class ComputeMethod(Enum):
    STANDARD = auto()
    EFFICIENT = auto()


class EDType(Enum):
    SCALE_DEP = auto()
    GLOBAL = auto()
    LOCAL = auto()


class _EffectiveDimension(ABC):
    def __init__(self) -> None:
        super().__init__()

        self._beta: float = None

    @abstractmethod
    def _parameter_space_dim(self, eps: float = 0) -> float:
        """
        The dimension $d$ of the parameter space.

        Returns:
            float: The dimension of the parameter space
        """
        pass

    @abstractmethod
    def _compute_beta(self, eps: float = 0):
        """
        Compute the normalizing constant beta. 

        Returns:
            float: The normalizing constant beta.
        """
        pass

    def compute(self,
                n: Union[int, List[int]],
                ed_type: EDType,

                compute_method: ComputeMethod = ComputeMethod.EFFICIENT,
                gamma: float = 1,
                eps: float = 0,
                compute_beta: bool = True,
                chunk_size: int = 100,

                verbose: bool = True,
                ) -> NDArray[np.float64]:
        """
        Computes the effective dimension of the model using the specified methods.

        Args:
            n (Union[int, List[int]]): The number of observations or a list of numbers of observations.
            ed_type (EDType): The type of effective dimension to compute (scale_dependent, global, or local).
            compute_method (ComputeMethod, optional): The computation method to use (standard or efficient). Default is ComputeMethod.EFFICIENT.
            gamma (float, optional): The constant gamma for the computation. Default is 1.
            eps (float, optional): The ball radius for local effective dimension. Required if ed_type is LOCAL. Default is 0.
            compute_beta (bool, optional): Specify whether the variable beta should be computed. Default is True.
            chunk_size (int, optional): The chunk size for computing effective dimensions for multiple ns. Default is 100.
            verbose (bool, optional): If True, enables verbose output. Default is True.

        Returns:
            NDArray[np.float64]: The computed effective dimension.
        """
        assert ed_type != EDType.LOCAL or eps > 0, "eps must be greater 0 if ed_type=LOCAL"
        assert compute_beta or self._beta, "beta was None although compute_beta=False"

        if isinstance(n, int):
            n = [n]
        n: NDArray[np.int32] = np.array(n).reshape(-1).astype(int)

        if compute_beta:
            self._beta = self._compute_beta(eps)

        kappa = n / (2*np.pi)

        if ed_type == EDType.SCALE_DEP:
            pass
        elif ed_type == EDType.GLOBAL or ed_type == EDType.LOCAL:
            kappa = gamma*n / (2 * np.pi * np.log2(n))
        else:
            raise ValueError("Invalid EDType specified.")

        if compute_method == ComputeMethod.STANDARD:
            return self._compute_standard(kappa, verbose=verbose)
        elif compute_method == ComputeMethod.EFFICIENT:
            return self._compute_efficient(
                kappa=kappa,
                eps=eps,
                chunk_size=chunk_size,
                verbose=verbose,
            )
        else:
            raise ValueError("Invalid EDType specified.")

    @abstractmethod
    def _compute_standard(self, kappa: float, verbose: bool = True) -> float:
        """
        Computes the effective dimension of the model using the specified method.

        Args:
            kappa (float): The constant kappa for the computation.
            verbose (bool, optional): If True, enables verbose output. Default is True.

        Returns:
            float: The computed effective dimension.
        """
        pass

    @abstractmethod
    def _compute_efficient(self, kappa: float, eps: float = 0, chunk_size: int = 100, verbose: bool = True) -> float:
        """
        Computes the effective dimension of the model using the refined formula.

        Args:
            kappa (float): The constant kappa for the computation.
            eps (float, optional): The ball radius for local effective dimension. Default is 0.
            chunk_size (int, optional): The chunk size for computing effective dimensions for multiple values of kappa. Default is 100.
            verbose (bool, optional): If True, enables verbose output. Default is True.

        Returns:
            float: The computed effective dimension.
        """
        pass


class EffectiveDimensionExact(_EffectiveDimension):
    def __init__(self) -> None:
        self._T = self._trace_integral()
        self._T_local: float = None

        super().__init__()

    @abstractmethod
    def _parameter_space_volume(self, eps: float = 0) -> float:
        """
        The volume $V_{\Theta}$ of the parameter.

        This function should not be called for Neural Network as it potentially 
        causes underflow or overflow.

        Args:
        eps: float, optional
            The ball radius for the local effective dimension

        Returns:
        float: The volume of the parameter space
        """
        pass

    @abstractmethod
    def _trace_integral(self, eps: float = 0) -> float:
        """
        Compute the integral of trace of package.fim.

        Return:
        float: integral value.
        """
        pass

    @abstractmethod
    def _numerator_integral(self, kappa: float) -> float:
        """
        Compute the numerator's integral. 

        This method is specific for the standard computation. 

        Args:
        kappa: kappa_{gamma, n}

        Returns:
        float: integral value
        """
        pass

    @abstractmethod
    def _zmax_and_zintegral(self, kappa: float, eps: float = 0) -> tuple[float, float]:
        """
        Compute the maximal value of z(theta) across the parameter space and 
        the numerator's integral. 

        This method is specific for the efficient computation. 

        Args:
        kappa: float
            kappa_{gamma, n}
        eps: float, optional
            The ball radius for local effective dimension

        Returns:
        float: maximal value of z(theta)
        float: integral value
        """
        pass

    def _compute_standard(self, kappa: float) -> float:
        return 2 * np.log2(self._numerator_integral(kappa)/self._parameter_space_volume()) \
            / np.log2(kappa)

    def _compute_efficient(self, kappa: float, eps: float = 0) -> float:
        zmax, integral = self._zmax_and_zintegral(kappa, eps)

        if eps > 0:
            volume = self._parameter_space_volume(eps)
        else:
            volume = self._parameter_space_volume()

        return 2 * zmax / np.log2(kappa) + 2 / np.log2(kappa) \
            * np.log2(integral/volume)


class EffectiveDimensionApprox(_EffectiveDimension):
    def __init__(self,
                 file_paths: Union[str, List[str]],
                 local_file_paths: Union[str, List[str]] = [],
                 cache_eigenvalues: bool = True
                 ) -> None:
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        if isinstance(local_file_paths, str):
            local_file_paths = [local_file_paths]

        assert os.path.isfile(
            file_paths[0]), f"file not found: {file_paths[0]}"

        self._file_paths = file_paths
        self._local_file_paths = local_file_paths

        with h5py.File(file_paths[0], 'r') as f:
            theta = f['thetas'][0]

            self._dim = len(theta)

        self._eigenvalues = None
        self._eigenvalues_local = None
        self._cache_eigenvalues = cache_eigenvalues

        super().__init__()

    def _parameter_space_dim(self) -> float:
        return self._dim

    def _mean_trace(self, eps: float = 0) -> float:
        """
        Compute the mean of trace of the Fisher Information Matrices

        Args:
            eps (float, optional): The ball radius for the local effective dimension

        Returns:
            float: The mean of trace of the Fisher Information Matrices
        """
        if eps > 0:
            assert os.path.isfile(
                self._local_file_paths[0]), f"file not found: {self._local_file_paths[0]}"
            file_paths = self._local_file_paths
        else:
            file_paths = self._file_paths

        mean_trace = 0
        for file_path in file_paths:
            with h5py.File(file_path, 'r') as f:
                max_ind = len(f['thetas'][:])

                traces = np.array([
                    sum(f[f'eigenvalues/index_{index}'][:])
                    for index in range(max_ind)
                ])

            mean_trace += np.mean(traces)

            del traces

        return mean_trace / len(file_paths)

    def _compute_beta(self, eps: float = 0):
        return self._parameter_space_dim() / self._mean_trace(eps)

    def compute(self,
                n: Union[int, List[int]],
                ed_type: EDType,
                compute_method: ComputeMethod = ComputeMethod.EFFICIENT,
                gamma: float = 1,
                eps: float = 0,
                compute_beta: bool = True,
                chunk_size: int = 100,
                verbose: bool = True
                ) -> NDArray[np.float64]:
        assert compute_method == ComputeMethod.EFFICIENT, "Neural network models must use the 'EFFICIENT' compute method."
        return super().compute(n, ed_type, compute_method, gamma, eps, compute_beta, chunk_size, verbose)

    def _compute_standard(self, kappa: float) -> float:
        # This function is not for NN
        return super()._compute_standard(kappa)

    def _compute_efficient(self, kappa: NDArray[np.float64], eps: float = 0, chunk_size: int = 100, verbose: bool = True) -> NDArray[np.float64]:
        """
        Computes the effective dimension of the model using the efficient method.

        Args:
            kappa (NDArray[np.float64]): The array of kappa values for the computation.
            eps (float, optional): The ball radius for local effective dimension. Default is 0.
            chunk_size (int, optional): The chunk size for computing effective dimensions for multiple values of kappa. Default is 100.
            verbose (bool, optional): If True, enables verbose output. Default is True.

        Returns:
            NDArray[np.float64]: The computed effective dimensions for each kappa value.
        """
        file_paths = self._local_file_paths if eps > 0 else self._file_paths
        results = None

        for file_path in file_paths:
            temp_eigenvalues = None
            temp_results = []

            if self._cache_eigenvalues:
                if self._eigenvalues is None:
                    with h5py.File(file_path, 'r') as f:
                        max_ind = len(f['thetas'][:])
                        self._eigenvalues = np.array(
                            [f[f'eigenvalues/index_{index}'][:] for index in range(max_ind)])
            else:
                with h5py.File(file_path, 'r') as f:
                    max_ind = len(f['thetas'][:])
                    temp_eigenvalues = np.array(
                        [f[f'eigenvalues/index_{index}'][:] for index in range(max_ind)])

            total_chunks = (len(kappa) + chunk_size - 1) // chunk_size
            if verbose:
                progress_bar = tqdm(total=total_chunks,
                                    desc=f'Computing EDs with chunk_size={chunk_size}')

            for start in range(0, len(kappa), chunk_size):
                end = start + chunk_size
                kappa_chunk = kappa[start:end]

                eigenvalues = temp_eigenvalues if not self._cache_eigenvalues else self._eigenvalues

                zs = np.sum(np.log2(
                    1 + self._beta * kappa_chunk[:, None, None] * eigenvalues[None, :, :]), axis=2) / 2

                zmax = np.max(zs, axis=1)

                result_chunk = 2 * \
                    (zmax + np.log2(np.mean(np.power(2, zs.T - zmax), axis=0))) / \
                    np.log2(kappa_chunk)

                temp_results.append(result_chunk)

                if verbose:
                    progress_bar.update(1)

            if verbose:
                progress_bar.close()

            if not self._cache_eigenvalues:
                del temp_eigenvalues

            if results is None:
                results = np.concatenate(temp_results)
            else:
                results += np.concatenate(temp_results)

        return results / len(file_paths)
