import numpy as np
from scipy.integrate import quad

from .effective_dimension import EffectiveDimensionExact


class Poisson(EffectiveDimensionExact):
    def __init__(self, theta_min: float, theta_max: float) -> None:
        assert theta_min < theta_max

        self._theta_min = theta_min
        self._theta_max = theta_max

        super().__init__()

    def _parameter_space_volume(self) -> float:
        """
        The volume $V_{\Theta}$ of the parameter.

        Returns:
        float: The volume of the parameter space
        """
        return self._theta_max - self._theta_min

    def _parameter_space_dim(self) -> float:
        return 1

    def _trace_integral(self):
        return quad(
            lambda x: 1/x,
            self._theta_min,
            self._theta_max,
        )[0]
    
    def _numerator_integral(self, kappa: float) -> float:
        return quad(
            lambda x: np.sqrt(1 + self._beta * kappa / x),
            self._theta_min,
            self._theta_max
        )[0]
        

    def _zmax_and_zintegral(self, kappa: float) -> tuple[float, float]:
        zmax = np.log2(1 + self._beta*kappa / self._theta_min)/2

        integral = quad(
            lambda x: 2**(np.log2(1+self._beta*kappa/x)/2 - zmax),
            self._theta_min,
            self._theta_max,
        )[0]

        return zmax, integral