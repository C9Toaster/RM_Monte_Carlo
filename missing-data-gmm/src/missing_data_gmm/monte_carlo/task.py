"""Script running replication of Monte Carlo simulation."""

from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import task

from missing_data_gmm.config import (
    DATA_CATALOGS,
    MC_DESIGNS,
    METHODS,
    MISSINGNESS_OPTIONS,
)
from missing_data_gmm.monte_carlo.helper import (
    apply_method,
    generate_data,
    initialize_replication_params,
    results_statistics,
)


def _error_handling_methods(methods: list):
    for method in methods:
        if method not in METHODS:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)


def _error_handling_params(params: dict):
    if not all(
        isinstance(params[key], int)
        for key in [
            "n_observations",
            "k_regressors",
            "n_replications",
            "n_complete",
            "n_missing",
            "random_key",
        ]
    ):
        msg = """Parameters n_observations, k_regressors, n_replications, n_complete,
        n_missing, and random_key must be integers."""
        raise ValueError(msg)
    if not all(isinstance(params[key], (float)) for key in ["lambda_"]):
        msg = "Parameter lambda_ must be a float."
        raise ValueError(msg)
    if not all(
        isinstance(params[key], np.ndarray)
        for key in [
            "alpha_coefficients",
            "b0_coefficients",
            "delta_coefficients",
            "gamma_coefficients",
            "theta_coefficients",
        ]
    ):
        msg = """Parameters alpha_coefficients, b0_coefficients, delta_coefficients,
        gamma_coefficients, and theta_coefficients must be numpy arrays."""
        raise ValueError(msg)
    if not isinstance(params["methods"], list):
        msg = "Parameter methods must be a list."
        raise TypeError(msg)
    if not all(isinstance(key, str) for key in params["methods"]):
        msg = "All elements in methods must be strings."
        raise ValueError(msg)


def _error_handling(params: dict):
    _error_handling_methods(params["methods"])
    _error_handling_params(params)


for design in MC_DESIGNS:
    for missingness in MISSINGNESS_OPTIONS:
        params = initialize_replication_params(design, missingness)

        @task(id=f"{design}_{missingness}")
        @pytask.mark.wip
        def task_simulate(
            params: Annotated[dict, params],
        ) -> Annotated[
            pd.DataFrame,
            DATA_CATALOGS["simulation"][f"MC_RESULTS_{design}_{missingness}"],
        ]:
            """Run Monte Carlo simulation for different methods and missingness options.

            Parameters:
            params (dict): Simulation parameters.
            random_key (int): Random seed for reproducibility.
            missingness (str): Missingness option (MCAR, MAR, MNAR, MNAR_shift).

            Returns:
            pd.DataFrame: Formatted simulation results.
            """
            _error_handling(params)
            rng = np.random.default_rng(params["random_key"])
            results = {method: [] for method in params["methods"]}
            for _ in range(params["n_replications"]):
                data = generate_data(params, rng)
                for method in params["methods"]:
                    results[method].append(apply_method(data, method, params))
            return results_statistics(results, params)
