from dataclasses import dataclass, field
import numpy as np


class DataConfig:
    """Configuration for synthetic data generation."""
    n: int = 1000
    phi_true: float = 1.0
    gamma_true: float = 1.0
    sigma2_eta_true: float = 0.1
    Sigma_true: np.ndarray = np.array([[0.01, 0.0], [0.0, 0.01]])
    sigma_p_true: np.ndarray = np.array([[0.015, 0.0], [0.0, 0.015]])
    color_z: str = '#386cb0'
    color_a0: str = '#f0027f'
    color_a1: str = '#bf5b17'

class ModelConfig:
    """Configuration parameters for the TVP model (used only in experiments)."""
    num_iters: int = 2000
    burn_in: int = 1800
    mu0: float = 10000.0
    sigma_init: np.ndarray = np.array([[0.015, 0.0], [0.0, 0.015]])
    sigma2_eta_init: float = 0.05
    gamma_init: float = 1.0
    phi_init: float = 1.0
    sample_phi: bool = False
    sample_gamma: bool = False
    alpha_phi0: float = 0.5
    beta_phi0: float = 0.5
    nu0_sigma2_eta: float = 2.0
    V0_sigma2_eta: float = 1.0
    gamma_0: float = 2.1
    V0_gamma: float = 0.5
    seed: int | None = 25

class ExperimentConfig:
    """Overall experiment configuration."""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    seed: int = 25
