import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


def plot_comparison(
    true_corr: np.ndarray,
    tvp_corr: np.ndarray,
    dcc_results: Dict[str, Any],
    save_path: str = None,
) -> plt.Figure:
    """Create a comparison plot of all methods."""
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(true_corr, label='True correlation', color='black', linewidth=3, alpha=0.7)
    ax.plot(tvp_corr, label='TVP Algorithm', color='blue', linewidth=2)

    classical = dcc_results['classical']['dcc_corr']
    t_copula = dcc_results['t_copula']['dcc_corr']

    ax.plot(classical, label='Classical DCC-GARCH', color='red', linestyle='--', linewidth=1.5)
    ax.plot(t_copula, label='DCC-GARCH (t-copula)', color='green', linestyle='-.', linewidth=1.5)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Comparison of TVP Algorithm with DCC-GARCH Methods', fontsize=16)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
