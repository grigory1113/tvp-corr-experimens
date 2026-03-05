import numpy as np

class OFZRubleConfig:
    """Configuration for OFZ and ruble experiment"""
    
    # Data parameters
    DATA_FREQUENCY = 'Q'  # 'Q', 'M', 'W'
    ESTIMATE_X_VARIANCE_METHOD = 'garch'  # 'garch', 'rolling', 'ewma'
    
    # GARCH parameters for x variance estimation
    GARCH_PARAMS = {
        'p': 1,
        'q': 0,
        'mean': 'constant',
        'dist': 'normal'
    }
    
    # Rolling window parameters for correlation
    ROLLING_WINDOW = 30
    
    # MCMC parameters
    MCMC_ITERATIONS = 2000
    BURN_IN_RATIO = 0.9  # 90% for burn-in
    
    # Prior distributions (tuned for OFZ-ruble data)
    class PriorParams:
        # For Sigma (transition covariance for a)
        mu0 = 10000
        nu0 = 0.5
        V0 = 0.5
        
        # For sigma2_eta
        nu0_sigma2_eta = 2
        V0_sigma2_eta = 1
        
        # For phi (if we sample it)
        alpha_phi0 = 0.5
        beta_phi0 = 0.5
        
        # For gamma (if we sample it)
        gamma_0 = 2.1
        V0_gamma = 0.5
        
        # Initial values
        sigma_init = np.array([[0.015, 0], [0, 0.015]])
        sigma2_eta_init = 0.1
        phi_init = 1.0 
        gamma_init = 1.0
        
        # Sampling flags
        sample_phi = False
        sample_gamma = False
    
    # Colors for plots
    COLORS = {
        'ofz': '#FFD700',          # OFZ
        'ruble': '#f0027f',          # Raspberry
        'correlation_tvp': '#bf5b17',  # Orange
        'correlation_dcc': '#7fc97f',  # Green
        'correlation_rolling': '#beaed4',  # Purple
        'confidence_interval': 'lightblue'
    }
