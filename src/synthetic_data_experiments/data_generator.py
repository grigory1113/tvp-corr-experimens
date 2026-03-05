import numpy as np
import matplotlib.pyplot as plt


def generate_data(n, phi_true, gamma_true, sigma2_eta_true, Sigma_true, sigma_p_true, seed=25):
    """Generate synthetic data for TVP model with explicit parameters."""
    np.random.seed(seed)

    z = np.zeros([n, 1])
    x_raw = np.random.normal(0, 1, n).reshape(n, 1)
    x = np.concatenate((np.ones([n, 1]), x_raw), axis=1)
    a = np.zeros([n, 2])
    h = np.zeros(n)
    sigma2_t = np.zeros(n)

    h[0] = 0
    a[0, :] = 0.2
    sigma2_t[0] = 1

    for i in range(1, n):
        h[i] = phi_true * h[i - 1] + np.random.normal(0, sigma2_eta_true)
        sigma2_t[i] = gamma_true * np.exp(h[i])

    for i in range(1, n):
        a[i, :] = a[i - 1, :] + np.random.multivariate_normal([0, 0], Sigma_true, 1)

    for i in range(n):
        z[i, 0] = a[i, 0] + a[i, 1] * x[i, 1] + np.random.normal(0, sigma2_t[i] ** 0.5)

    phro = a[:, 1] * 1 / (a[:, 1] ** 2 * 1 + sigma2_t) ** 0.5

    return {
        'z': z,
        'x': x,
        'a_true': a,
        'h_true': h,
        'sigma2_t_true': sigma2_t,
        'phro_true': phro,
        'params': {
            'phi': phi_true,
            'gamma': gamma_true,
            'sigma2_eta': sigma2_eta_true,
            'Sigma': Sigma_true
        }
    }


def plot_generated_data(z, a, colors=None):
    """Visualize generated data."""
    if colors is None:
        colors = {'z': '#386cb0', 'a0': '#f0027f', 'a1': '#bf5b17'}

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(z, label='z', color=colors['z'], alpha=0.7)
    ax.plot(a[:, 0], label='a$_0$', color=colors['a0'], alpha=0.7)
    ax.plot(a[:, 1], label='a$_1$', color=colors['a1'], alpha=0.7)

    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Generated Data', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#444444')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
