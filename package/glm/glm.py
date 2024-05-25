import os
import numpy as np
from numpy.typing import NDArray
from scipy.special import xlogy
from typing import List, Tuple
from pandas import DataFrame
import patsy
from sklearn.preprocessing import MinMaxScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper, GLM
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import chi2

from package.effective_dimension.effective_dimension import EDType, EffectiveDimensionApprox
from package.fim.fim import compute_fims_glm
from package.directories import eigenvalues_dir


def deviance_loss(beta: NDArray, x: NDArray, y: NDArray, batch_size: int = 3000):
    """
    Computes the deviance loss for a given set of parameters, features, and response variables.

    Args:
        beta (NDArray): The parameter values for the GLM.
        x (NDArray): The input features for the GLM.
        y (NDArray): The response variable for the GLM.
        batch_size (int, optional): The number of samples per batch for computing the loss. Default is 3000.

    Returns:
        float: The computed deviance loss.
    """
    n_samples = x.shape[0]
    # Calculate the number of batches
    num_batches = (n_samples + batch_size - 1) // batch_size
    total_loss = 0.0

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples)

        x_batch = x[start:end]
        y_batch = y[start:end]

        predicted = np.exp(x_batch @ beta)
        deviance = 2 * (xlogy(y_batch, y_batch / predicted) -
                        y_batch + predicted)

        total_loss += np.sum(deviance)

    return total_loss / n_samples


def compute_temp_eff_dim(
        df: DataFrame,
        formula: str,
        num_betas: int,
        batch_size: int=0,
        eps: float=0,
        gamma=1,
        beta_center: NDArray=None,

        con_min: float=0,
        con_max: float=np.inf,
):
    """
    Computes the temporary effective dimension of the Poisson Generalized Linear Model (GLM) using a given dataset and formula.

    Args:
        df (DataFrame): The input DataFrame containing the dataset.
        formula (str): The formula specifying the model.
        num_betas (int): The number of beta values to generate.
        batch_size (int, optional): The number of samples per batch for FIM computation. Default is 0.
        eps (float, optional): Radius for local effective dimension. Default is 0.
        gamma (float, optional): The gamma parameter for scaling. Default is 1.
        beta_center (NDArray, optional): The central value for beta around which to generate samples. Default is None.
        con_min (float, optional): The minimum constraint on exp(<beta, X>). Default is 0.
        con_max (float, optional): The maximum constraint on exp(<beta, X>). Default is np.inf.
    Returns:
        float: The computed effective dimension.
    """
    _, x = patsy.dmatrices(formula, df, return_type='matrix')

    if eps == 0:
        compute_fims_glm(
            x.shape[1],
            num_betas,
            features=x,
            save_dir=os.path.join(eigenvalues_dir, 'temp'),
            filename='temp',
            verbose=False,
            batch_size=batch_size,
            con_min=con_min,
            con_max=con_max
        )
    else:
        assert (isinstance(eps, float) or isinstance(eps, int)) and eps > 0
        assert beta_center is not None

        compute_fims_glm(
            x.shape[1],
            num_betas,
            features=x,
            save_dir=os.path.join(eigenvalues_dir, 'temp'),
            filename='temp',
            verbose=False,
            batch_size=batch_size,
            beta_center=beta_center,
            beta_min=-eps,
            beta_max=eps,
            con_min=con_min,
            con_max=con_max
        )

    file_path = os.path.join(
        eigenvalues_dir, 'temp', f'temp_{x.shape[1]}.h5')
    ef = EffectiveDimensionApprox(file_path, file_path)

    return ef.compute(len(df), EDType.GLOBAL, verbose=False, gamma=gamma, eps=eps)


def backward_wo_con(
    train: DataFrame,
    formula_full: str,

    alpha: float = 0.05,
) -> List[GLMResultsWrapper]:
    """
    Performs backward elimination for the Poisson Generalized Linear Model (GLM) without constraints.

    Args:
        train (DataFrame): The training dataset.
        formula_full (str): The full model formula.
        alpha (float, optional): The significance level for retaining a feature. Default is 0.05.

    Returns:
        List[GLMResultsWrapper]: A list of GLM results from each step of the backward elimination process.
    """
    results = []

    model = smf.glm(formula=formula_full, data=train,
                    family=sm.families.Poisson())

    result = model.fit(
        np.zeros(model.exog.shape[1],), method="newton", maxiter=100)

    results.append(result)

    while True:
        p_values = result.pvalues
        max_p_value = p_values.max()

        if max_p_value > alpha:
            excluded_variable = p_values.idxmax()
            print(f"Removing {excluded_variable} with p-value {max_p_value}")

            formula = result.model.formula
            lhs, rhs = formula.split('~')
            rhs_terms = rhs.strip().split(' + ')

            if 'C(' in excluded_variable:
                base_term = excluded_variable.split('[')[0]
                rhs_terms = [
                    term for term in rhs_terms if not term.startswith(base_term)]
            else:
                rhs_terms = [
                    term for term in rhs_terms if term != excluded_variable]

            new_formula = f'{lhs.strip()} ~ {" + ".join(rhs_terms)}'

            print(new_formula)

            model = smf.glm(formula=new_formula, data=train,
                            family=sm.families.Poisson())
            result = model.fit(
                np.zeros(model.exog.shape[1],), method="newton", maxiter=100)

            results.append(result)
        else:
            break

    return results


def fit_wo_con(
        train: DataFrame,
        formula: str,
        max_iter: int=100
) -> Tuple[GLM, List[NDArray], GLMResultsWrapper]:
    """
    Fits the Poisson Generalized Linear Model (GLM) without constraints using the provided training data and formula.

    Args:
        train (DataFrame): The training dataset.
        formula (str): The model formula.
        max_iter (int, Optional): The maximum number of iteration. Default is 100.

    Returns:
        Tuple[GLM, List[NDArray], GLMResultsWrapper]:
            - model: The fitted GLM model.
            - parameters: List of parameters during training.
            - result: The GLM results wrapper containing the fitted model results.
    """
    parameters = []

    def save_param(params):
        parameters.append(params.copy())
        return False

    model = smf.glm(formula=formula,
                    data=train, family=sm.families.Poisson())
    
    initial_beta = np.zeros(model.exog.shape[1])
    parameters.append(initial_beta)

    result = model.fit(initial_beta, method="newton", maxiter=max_iter, callback=save_param)

    return model, parameters, result


def fit_w_con(
    train: DataFrame,
    formula: str,
    con_min: float,
    con_max: float,

    batch_size: int = 3000,
) -> Tuple[NDArray, NDArray, NDArray, any]:
    """
    Fits the Poisson Generalized Linear Model (GLM) with constraints using the provided training data and formula.

    Args:
        train (DataFrame): The training dataset.
        formula (str): The model formula.
        con_min (float): The minimum constraint on exp(<beta, X>).
        con_max (float): The maximum constraint on exp(<beta, X>).
        batch_size (int, optional): The number of samples per batch for computing the loss. Default is 3000.

    Returns:
        Tuple[NDArray, NDArray, NDArray, any]:
            - X: The input features used in the model.
            - y: The response variable used in the model.
            - beta: The fitted parameter values.
            - result: The result object from the optimizer.
    """
    y, x = patsy.dmatrices(formula, train, return_type='matrix')
    y = y.reshape(-1)

    parameters = []

    def loss(beta: NDArray, x: NDArray, y: NDArray):
        parameters.append(beta)
        return deviance_loss(beta, x, y, batch_size)

    def constraint(beta: NDArray, x: NDArray):
        mu = np.exp(x @ beta)
        return np.array([con_max - mu.min(), mu.max() - con_min])

    initial_beta = np.zeros(x.shape[1])

    parameters.append(initial_beta)

    beta_bounds = [(-1, 0) for _ in range(x.shape[1])]

    result = minimize(loss, initial_beta, args=(x, y), method='SLSQP',
                      constraints=[
                          {'type': 'ineq', 'fun': constraint, 'args': (x,)}],
                      bounds=beta_bounds)

    return x, y, parameters, result
