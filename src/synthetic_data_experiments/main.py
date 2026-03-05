#!/usr/bin/env python3
"""Main script to compare TVP algorithm with DCC-GARCH methods."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from tvp_correlation.main import Model
from .config import ExperimentConfig
from .data_generator import generate_data, plot_generated_data
from src.utils.dcc_garch import DCCGARCH
from .plot_results import plot_comparison


def calculate_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    min_len = min(len(true), len(pred))
    true = true[:min_len]
    pred = pred[:min_len]
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - true))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae}


def main():
    exp_config = ExperimentConfig()
    np.random.seed(exp_config.seed)

    # 1. Generate data
    print("Generating synthetic data...")
    data = generate_data(
        n=exp_config.data.n,
        phi_true=exp_config.data.phi_true,
        gamma_true=exp_config.data.gamma_true,
        sigma2_eta_true=exp_config.data.sigma2_eta_true,
        Sigma_true=exp_config.data.Sigma_true,
        sigma_p_true=exp_config.data.sigma_p_true,
        seed=exp_config.seed
    )

    # Optional: plot generated data
    colors = {'z': exp_config.data.color_z, 'a0': exp_config.data.color_a0, 'a1': exp_config.data.color_a1}
    fig_data = plot_generated_data(data['z'], data['a_true'], colors=colors)
    plt.savefig('results/generated_data.png', dpi=300, bbox_inches='tight')
    plt.close(fig_data)

    # 2. Run TVP algorithm
    print("Running TVP algorithm...")
    tvp_model = Model(
        num_iters=exp_config.model.num_iters,
        burn_in=exp_config.model.burn_in,
        mu0=exp_config.model.mu0,
        sigma_init=exp_config.model.sigma_init,
        sigma2_eta_init=exp_config.model.sigma2_eta_init,
        gamma_init=exp_config.model.gamma_init,
        phi_init=exp_config.model.phi_init,
        sample_phi=exp_config.model.sample_phi,
        sample_gamma=exp_config.model.sample_gamma,
        alpha_phi0=exp_config.model.alpha_phi0,
        beta_phi0=exp_config.model.beta_phi0,
        nu0_sigma2_eta=exp_config.model.nu0_sigma2_eta,
        V0_sigma2_eta=exp_config.model.V0_sigma2_eta,
        gamma_0=exp_config.model.gamma_0,
        V0_gamma=exp_config.model.V0_gamma,
        seed=exp_config.seed
    )
    tvp_results = tvp_model.run(data['z'], data['x'], progress_bar=True)
    tvp_corr, _, _ = tvp_model.compute_correlation(tvp_results)  # теперь метод класса

    # 3. DCC-GARCH
    print("Estimating DCC-GARCH models...")
    x1 = data['x'][:, 1]
    z1 = data['z'].flatten()
    rets = pd.DataFrame({'x': x1, 'z': z1})

    dcc = DCCGARCH()
    dcc_results = dcc.fit_both_models(rets)

    # 4. Plot comparison
    print("Creating comparison plot...")
    fig = plot_comparison(
        data['phro_true'],
        tvp_corr,
        dcc_results,
        save_path="results/comparison.png"
    )
    plt.show()

    # 5. Metrics
    print("\nQuality metrics:")
    metrics = {}
    metrics['TVP'] = calculate_metrics(data['phro_true'], tvp_corr)
    metrics['Classical DCC'] = calculate_metrics(data['phro_true'], dcc_results['classical']['dcc_corr'])
    metrics['t-copula DCC'] = calculate_metrics(data['phro_true'], dcc_results['t_copula']['dcc_corr'])

    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df.round(6))

    # 6. Save results
    os.makedirs('results', exist_ok=True)
    metrics_df.to_csv('results/metrics.csv')
    with open('results/dcc_parameters.json', 'w') as f:
        dcc_params = {
            'classical': dcc_results['classical']['params'].tolist(),
            't_copula': dcc_results['t_copula']['params'].tolist(),
        }
        json.dump(dcc_params, f, indent=2)

    print("\nResults saved in 'results/' folder.")


if __name__ == "__main__":
    main()
