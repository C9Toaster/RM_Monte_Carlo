"""Helper functions for the Monte Carlo simulation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from missing_data_gmm.config import METHODS
from missing_data_gmm.monte_carlo.complete import complete_case_method
from missing_data_gmm.monte_carlo.dagenais import dagenais_weighted_method
from missing_data_gmm.monte_carlo.dummy import dummy_variable_method
from missing_data_gmm.monte_carlo.gmm import gmm_method


def _get_design_parameters(design: int) -> list:
    match design:
        case 0:
            return [np.array([1]), np.array([1, 1, 1]), np.array([1, 1]), False]
        # case 0 is equivalent to case 5
        case 1:
            return [np.array([1]), np.array([10, 0, 0]), np.array([10, 0]), False]
        case 2:
            return [np.array([0.1]), np.array([10, 0, 0]), np.array([10, 0]), False]
        case 3:
            return [np.array([1]), np.array([1, 0, 0]), np.array([10, 0]), False]
        case 4:
            return [np.array([1]), np.array([1, 0, 1]), np.array([1, 1]), False]
        case 5:
            return [np.array([1]), np.array([1, 1, 1]), np.array([1, 1]), False]
        case 6:
            return [np.array([1]), np.array([1, 0, 0]), np.array([1, 0]), False]
        case 7:
            return [np.array([1]), np.array([1, 0, 1]), np.array([1, 1]), False]
        case 8:
            return [
                np.array([1]),
                np.array([0.1, 0.2, 0.1]),
                np.array([0.1, 0.2]),
                True,
            ]


def initialize_replication_params(design: int = 0, missingness: str = "MCAR") -> dict:
    """Initialize parameters for the Monte Carlo simulation.

    Returns:
        dict: Parameters for the Monte Carlo simulation.
    """
    params = {}
    params["design"] = design
    params["missingness"] = missingness
    params["methods"] = METHODS
    params["n_observations"] = 400  # 5000  # Number of observations
    params["k_regressors"] = 3  # Number of regressors (including intercept)
    params["lambda_"] = 0.5  # Proportion of observations with missing data
    params["n_replications"] = 5000  # Number of Monte Carlo replications
    params["n_complete"] = int(
        params["n_observations"] * params["lambda_"]
    )  # Number of complete cases
    params["n_missing"] = (
        params["n_observations"] - params["n_complete"]
    )  # Number of missing cases

    keys = [
        "alpha_coefficients",
        "theta_coefficients",
        "delta_coefficients",
        "exponential",
    ]
    values = _get_design_parameters(design)
    params.update(dict(zip(keys, values, strict=False)))

    params["b0_coefficients"] = np.array(
        [params["alpha_coefficients"][0]] + [1] * (params["k_regressors"] - 1)
    )  # True coefficients
    params["gamma_coefficients"] = np.array(
        [1] * (params["k_regressors"] - 1)
    )  # Imputation coefficients

    params["max_iterations"] = 200  # number of max iterations of gmm
    params["random_key"] = 123456
    # params["random_key"] = 1234
    return params


def _generate_instruments(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate instrument variables z including intercept."""
    # binary_instrument = rng.standard_normal(n) > 0.5  # z1
    continuous_instrument = rng.standard_normal(n)  # z2
    return np.column_stack((np.ones(n), continuous_instrument))  # z with intercept


def _generate_x(
    z: np.ndarray,
    v: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Generate independent variable x with heteroskedasticity."""
    regressors = np.column_stack([np.ones_like(z[:, 1]), np.square(z[:, 1])])
    sigma_xi = np.sqrt(np.dot(regressors, params["delta_coefficients"]))
    mean_x = z @ params["gamma_coefficients"]  # mx

    if params["exponential"]:
        return mean_x + v * np.exp(sigma_xi)
    return mean_x + v * sigma_xi


def _generate_y(
    x: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Generate dependent variable y with heteroskedasticity."""
    mean_y = np.column_stack((x, z)) @ params["b0_coefficients"]  # my
    regressors = np.column_stack([np.ones_like(x), np.square(x), np.square(z[:, 1])])
    sigma_epsilon = np.sqrt(np.dot(regressors, params["theta_coefficients"]))

    if params["exponential"]:
        return mean_y + u * np.exp(sigma_epsilon)
    return mean_y + u * sigma_epsilon, u * sigma_epsilon


def _partition_data(x, z, y, n_complete, missing_indicator):
    """Partition data into complete and incomplete cases for MAR."""
    complete_cases = ~missing_indicator
    incomplete_cases = missing_indicator
    return {
        "w_complete": np.column_stack((x[complete_cases], z[complete_cases, :])),
        "x_complete": x[complete_cases],
        "y_complete": y[complete_cases],
        "z_complete": z[complete_cases, :],
        "w_missing": np.column_stack((x[incomplete_cases], z[incomplete_cases, :])),
        "x_missing": x[incomplete_cases],
        "y_missing": y[incomplete_cases],
        "z_missing": z[incomplete_cases, :],
        "n_complete": n_complete,
    }


def generate_data(params: dict, rng: np.random.Generator) -> dict:
    """Data generating process. This follows MAR assumption.

    Args:
        params (dict): Parameters of the Monte Carlo simulation.
        rng (np.random.Generator): Random number generator.

    Returns:
        dict: Generated data.
    """
    z = np.column_stack(
        (
            np.ones(params["n_observations"]),
            rng.standard_normal(params["n_observations"]),
        )
    )  # continuous instrument z with intercept

    u = rng.standard_normal(params["n_observations"])
    if params["design"] in [6, 7, 8]:
        v = np.square(u) - 1
    else:
        v = rng.standard_normal(params["n_observations"])
    x = _generate_x(z, v, params)

    if params["missingness"] == "MCAR":
        # Create missingness indicator where the first n_complete observations are complete
        missing_indicator = np.zeros(params["n_observations"], dtype=bool)
        missing_indicator[params["n_complete"] :] = True

    elif params["missingness"] == "MAR_z":
        # Create missingness indicator based on z
        prob_missing = 1 / (
            1 + np.exp(-0.5 * z[:, 1])
        )  # Logistic function of z2for missingness probability
        # prob_missing = norm.cdf(z @ params["gamma_coefficients"])
        missing_indicator = (
            rng.uniform(size=params["n_observations"]) < prob_missing
        )  # Missing indicator (1 = missing, 0 = observed)
        params["n_missing"] = np.sum(missing_indicator)
        params["n_complete"] = params["n_observations"] - params["n_missing"]

        # Ensure exactly params['n_missing'] observations in x are missing
        # n_missing = params["n_missing"]
        # if np.sum(missing_indicator) > n_missing:
        #     excess_missing = np.sum(missing_indicator) - n_missing
        #     missing_indices = np.where(missing_indicator)[0]
        #     keep_indices = rng.choice(missing_indices, excess_missing, replace=False)
        #     missing_indicator[keep_indices] = False
        # elif np.sum(missing_indicator) < n_missing:
        #     deficit_missing = n_missing - np.sum(missing_indicator)
        #     non_missing_indices = np.where(~missing_indicator)[0]
        #     add_indices = rng.choice(non_missing_indices, deficit_missing, replace=False)
        #     missing_indicator[add_indices] = True
        # x[missing_indicator] = np.nan

    elif params["missingness"] == "MNAR_shift":
        # Create missingness indicator where the first n_complete observations are complete
        missing_indicator = np.zeros(params["n_observations"], dtype=bool)
        missing_indicator[params["n_complete"] :] = True

        # Shift the missing x values by a constant
        x[missing_indicator] += 1.0

    y, epsilon = _generate_y(x, z, u, params)

    if params["missingness"] == "MAR_y":
        # Create missingness indicator based on y
        prob_missing = 1 / (
            1 + np.exp(-0.5 * y)
        )  # Logistic function of y for missingness probability
        missing_indicator = (
            rng.uniform(size=params["n_observations"]) < prob_missing
        )  # Missing indicator (1 = missing, 0 = observed)
        params["n_missing"] = np.sum(missing_indicator)
        params["n_complete"] = params["n_observations"] - params["n_missing"]

    partitions = _partition_data(x, z, y, params["n_complete"], missing_indicator)
    # reorder x,y,z to be have complete data first then missing data
    x = np.concatenate([partitions["x_complete"], partitions["x_missing"]])
    y = np.concatenate([partitions["y_complete"], partitions["y_missing"]])
    z = np.concatenate([partitions["z_complete"], partitions["z_missing"]])
    return {"x": x, "y": y, "z": z, "n_missing": params["n_missing"], **partitions}


def plot_missing_data(x, z, missing_indicator):
    """Visualize x and the second component of z in a scatterplot.

    Args:
        x (np.ndarray): Independent variable.
        z (np.ndarray): Instrument variables.
        missing_indicator (np.ndarray): Boolean array indicating missing data.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        x[~missing_indicator], z[~missing_indicator, 1], color="blue", label="Complete"
    )
    plt.scatter(
        x[missing_indicator], z[missing_indicator, 1], color="red", label="Missing"
    )
    plt.xlabel("x")
    plt.ylabel("z (second component)")
    plt.title("Scatterplot of x and z with Missing Data Indicator")
    plt.legend()
    plt.show()


def apply_method(data, method, params):
    """Apply the specified estimation method to the generated data.

    Parameters:
        data (dict): Generated data (from `generate_data`).
        method (str): The name of the estimation method to apply.
        params (dict): Simulation parameters.

    Returns:
        dict: Results containing estimates and standard errors.
    """
    if method == "Complete case method":
        return complete_case_method(data, params)
    if method == "Dummy case method":
        return dummy_variable_method(data, params)
    if method == "Dagenais (FGLS)":
        return dagenais_weighted_method(data, params)
    if method == "GMM":
        return gmm_method(data, params)
    msg = f"Unknown method: {method}"
    raise ValueError(msg)


def results_statistics(results: dict, params: dict) -> pd.DataFrame:
    """Calculate statistics of the results from the Monte Carlo simulation.

    Args:
        results (dict): Results from the Monte Carlo simulation.
        params (dict): Parameters of the Monte Carlo simulation.

    Returns:
        pd.DataFrame: DataFrame with statistics of the results.
    """
    parameters = [
        f"beta_{i}" if i > 0 else "alpha_0" for i in range(params["k_regressors"])
    ]
    results_df = []
    for method, method_results in results.items():
        coefficients = np.array([entry["coefficients"] for entry in method_results])

        mean_estimates = np.mean(coefficients, axis=0)
        mean_biases = mean_estimates - params["b0_coefficients"]
        n_vars = params["n_observations"] * np.var(coefficients, axis=0, ddof=1)
        mses = mean_biases**2 + (n_vars / params["n_observations"])

        # Create rows for the DataFrame
        for i, parameter in enumerate(parameters):
            results_df.append(
                {
                    "Method": method,
                    "Parameter": parameter,
                    "Bias": mean_biases[i],
                    "n*Var": n_vars[i],
                    "MSE": mses[i],
                }
            )

    return pd.DataFrame(results_df)
