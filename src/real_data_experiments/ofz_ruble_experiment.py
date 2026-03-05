import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

from src.real_data_experiments.config_ofz_ruble import OFZRubleConfig
from src.real_data_experiments.data_processor import DataProcessor
from tvp_correlation.main import Model
from src.utils.dcc_garch import DCCGARCH


class OFZRubleExperiment:
    """Experiment for estimating the correlation between OFZ and ruble"""
    
    def __init__(self, config):
        self.config = config
        self.data = None
        self.x_variance = None
        
    def load_and_prepare_data(self):
        """Load and prepare data, then filter from year 2000"""
        print("Loading OFZ and ruble data...")
        
        self.data = DataProcessor.load_ofz_and_ruble_data(
            ofz_path='data/ofz_data.csv',
            ruble_path='data/RC_F01_07_1992_T10_06_2025.xlsx',
            frequency=self.config.DATA_FREQUENCY
        )
        
        # Filter data from 2000 onwards
        start_date = getattr(self.config, 'START_DATE', '2000-01-01')
        dates = pd.to_datetime(self.data['dates'])
        mask = dates >= pd.Timestamp(start_date)
        
        for key in list(self.data.keys()):
            if key != 'params' and hasattr(self.data[key], '__len__') and len(self.data[key]) == len(dates):
                self.data[key] = self.data[key][mask]
        
        self.data['dates'] = dates[mask]
        
        # Estimate x variance
        print(f"Estimating x variance using {self.config.ESTIMATE_X_VARIANCE_METHOD} method...")
        
        if self.config.ESTIMATE_X_VARIANCE_METHOD == 'garch':
            self.x_variance, garch_results = DataProcessor.estimate_x_variance(
                self.data['x_raw'], 
                method='garch',
                model_params=self.config.GARCH_PARAMS
            )
            print(f"GARCH model parameters: {garch_results['model'].params}")
        else:
            self.x_variance, _ = DataProcessor.estimate_x_variance(
                self.data['x_raw'],
                method=self.config.ESTIMATE_X_VARIANCE_METHOD
            )
        
        self.data['x_variance'] = self.x_variance
        
        print(f"Number of observations: {len(self.data['z'])}")
        print(f"Period: {self.data['dates'][0]} - {self.data['dates'][-1]}")
        
        return self.data
    
    def get_model_params(self):
        """Return dictionary of parameters for TVPModel constructor."""
        prior = self.config.PriorParams
        return {
            'num_iters': self.config.MCMC_ITERATIONS,
            'burn_in': int(self.config.BURN_IN_RATIO * self.config.MCMC_ITERATIONS),
            'mu0': prior.mu0,
            'sigma_init': prior.sigma_init,
            'sigma2_eta_init': prior.sigma2_eta_init,
            'gamma_init': prior.gamma_init,
            'phi_init': prior.phi_init,
            'sample_phi': prior.sample_phi,
            'sample_gamma': prior.sample_gamma,
            'alpha_phi0': prior.alpha_phi0,
            'beta_phi0': prior.beta_phi0,
            'nu0_sigma2_eta': prior.nu0_sigma2_eta,
            'V0_sigma2_eta': prior.V0_sigma2_eta,
            'gamma_0': prior.gamma_0,
            'V0_gamma': prior.V0_gamma,
            'seed': 42
        }
    
    def run_tvp_algorithm(self):
        """Run TVP algorithm"""
        print("\nRunning TVP algorithm...")
        
        params = self.get_model_params()
        tvp_model = Model(**params)
        
        tvp_results = tvp_model.run(
            z=self.data['z'],
            x=self.data['x'],
            progress_bar=True
        )
        
        correlation, corr_low, corr_up = tvp_model.compute_correlation(
            results=tvp_results,
            burn_in=params['burn_in'],
            x_variance=self.x_variance
        )
        
        burn_in = params['burn_in']
        a_post = tvp_results['a_est'][burn_in:, :, :]
        a_mean = np.mean(a_post, axis=0)
        
        sigma_t_post = tvp_results['sigma_t_est_history'][burn_in:, :]
        sigma_t_mean = np.sqrt(np.mean(sigma_t_post, axis=0))
        
        z = self.data['z'].flatten()
        x_raw = self.data['x_raw'].flatten()
        residuals = z - (a_mean[:, 0] + a_mean[:, 1] * x_raw)
        standardized_residuals = residuals / sigma_t_mean
        
        tvp_results['standardized_residuals'] = standardized_residuals
        tvp_results['residuals'] = residuals
        tvp_results['sigma_t_mean'] = sigma_t_mean
        
        return {
            'results': tvp_results,
            'correlation': correlation,
            'corr_low': corr_low,
            'corr_up': corr_up,
            'model': tvp_model
        }
    
    def run_dcc_garch(self):
        """Run DCC-GARCH model"""
        print("\nRunning DCC-GARCH model...")
        
        rets = pd.DataFrame({
            'ofz': self.data['z'].flatten(),
            'ruble': self.data['x_raw'].flatten()
        }, index=self.data['dates'])
        
        dcc_model = DCCGARCH()
        dcc_results = dcc_model.fit_both_models(rets)
        
        return dcc_results
    
    def calculate_rolling_correlation(self):
        """Calculate rolling correlation"""
        print("\nCalculating rolling correlation...")
        
        rolling_corr = DataProcessor.calculate_rolling_correlation(
            self.data['z'],
            self.data['x_raw'],
            window=self.config.ROLLING_WINDOW
        )
        
        return rolling_corr
    
    def plot_results(self, tvp_results, dcc_results, rolling_corr):
        """Visualize results (two plots: original data and correlations)"""
        colors = self.config.COLORS

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # 1. Original data
        ax1 = axes[0]
        ax1.plot(self.data['dates'], self.data['ofz_growth'], 
                label='OFZ yield change, %', 
                color=colors['ofz'], alpha=0.7)
        ax1.plot(self.data['dates'], self.data['ruble_growth'], 
                label='Ruble exchange rate change, %', 
                color=colors['ruble'], alpha=0.7)
        ax1.set_ylabel('Change, %')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Dynamics of percentage changes')

        # 2. Correlations
        ax2 = axes[1]

        ax2.plot(self.data['dates'], tvp_results['correlation'], 
                label='Proposed Algorithm', 
                color=colors['correlation_tvp'], linewidth=2)
        ax2.fill_between(self.data['dates'], 
                        tvp_results['corr_low'], 
                        tvp_results['corr_up'], 
                        alpha=0.2, color=colors['confidence_interval'], 
                        label='95% CI (Proposed Algorithm)')

        dcc_corr = dcc_results['t_copula']['dcc_corr']
        ax2.plot(self.data['dates'], dcc_corr[:len(self.data['dates'])], 
                label='DCC-GARCH (t-copula)', 
                color=colors['correlation_dcc'], linestyle='--', linewidth=1.5)

        ax2.plot(self.data['dates'], rolling_corr, 
                label=f'Rolling correlation (window={self.config.ROLLING_WINDOW})', 
                color=colors['correlation_rolling'], linestyle=':', linewidth=1.5)

        ax2.set_ylabel('Correlation')
        ax2.set_xlabel('Date')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Dynamic correlation between OFZ yield and ruble exchange rate')
        ax2.set_ylim(-1.1, 1.1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # Shading crisis periods
        crisis_periods = [
            ('2008-09-01', '2009-03-01', 'Crisis 2008'),
            ('2014-12-01', '2015-06-01', 'Sanctions 2014'),
            ('2020-03-01', '2020-09-01', 'Crisis 2020'),
            ('2022-02-01', '2022-06-01', 'Sanctions 2022')
        ]

        for start, end, label in crisis_periods:
            try:
                ax2.axvspan(pd.Timestamp(start), pd.Timestamp(end), 
                        alpha=0.1, color='red')
            except:
                pass

        plt.tight_layout()
        return fig
    
    def plot_standardized_residuals(self, tvp_results):
        """Visualize standardized residuals"""
        colors = self.config.COLORS
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        ax1 = axes[0]
        standardized_residuals = tvp_results['results']['standardized_residuals']
        
        ax1.plot(self.data['dates'], standardized_residuals, 
                label='Standardized residuals', 
                color=colors['correlation_tvp'], alpha=0.7, linewidth=1)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax1.axhline(y=2, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='±2σ')
        ax1.axhline(y=-2, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ax1.set_ylabel('Standardized residuals')
        ax1.set_title('Standardized residuals: (z - a*x) / σ_t')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.hist(standardized_residuals, bins=30, alpha=0.7, 
                color=colors['correlation_tvp'], edgecolor='black')
        
        from scipy.stats import norm
        xmin, xmax = ax2.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, 0, 1)
        ax2.plot(x, p, 'k', linewidth=2, label='N(0,1)')
        
        ax2.set_xlabel('Standardized residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of standardized residuals')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def test_normality(self, standardized_residuals):
        """Normality tests for standardized residuals"""
        print("\n" + "=" * 70)
        print("NORMALITY TESTS FOR STANDARDIZED RESIDUALS")
        print("=" * 70)
        
        from scipy import stats
        
        mean_val = np.mean(standardized_residuals)
        std_val = np.std(standardized_residuals)
        skew_val = stats.skew(standardized_residuals)
        kurt_val = stats.kurtosis(standardized_residuals)
        
        print(f"Basic statistics:")
        print(f"  Mean: {mean_val:.4f} (expected: 0.000)")
        print(f"  Standard deviation: {std_val:.4f} (expected: 1.000)")
        print(f"  Skewness: {skew_val:.4f} (expected: 0.000)")
        print(f"  Kurtosis: {kurt_val:.4f} (expected: 0.000)")
        
        n = len(standardized_residuals)
        if n <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(standardized_residuals)
            print(f"\nShapiro-Wilk test:")
            print(f"  Statistic: {shapiro_stat:.4f}")
            print(f"  p-value: {shapiro_p:.4e}")
            print(f"  Normality: {'YES' if shapiro_p > 0.05 else 'NO'} (α=0.05)")
        
        jb_stat, jb_p = stats.jarque_bera(standardized_residuals)
        print(f"\nJarque-Bera test:")
        print(f"  Statistic: {jb_stat:.4f}")
        print(f"  p-value: {jb_p:.4e}")
        print(f"  Normality: {'YES' if jb_p > 0.05 else 'NO'} (α=0.05)")
        
        ks_stat, ks_p = stats.kstest(standardized_residuals, 'norm')
        print(f"\nKolmogorov-Smirnov test:")
        print(f"  Statistic: {ks_stat:.4f}")
        print(f"  p-value: {ks_p:.4e}")
        print(f"  Normality: {'YES' if ks_p > 0.05 else 'NO'} (α=0.05)")
        
        outside_2sigma = np.sum(np.abs(standardized_residuals) > 2) / n * 100
        outside_3sigma = np.sum(np.abs(standardized_residuals) > 3) / n * 100
        
        print(f"\nEmpirical rule check:")
        print(f"  Observations outside ±2σ: {outside_2sigma:.2f}% (expected: ~5%)")
        print(f"  Observations outside ±3σ: {outside_3sigma:.2f}% (expected: ~0.3%)")
        
        from scipy.stats import probplot
        _, (slope, intercept, r) = probplot(standardized_residuals, dist="norm", plot=None, fit=True)
        print(f"\nQ-Q plot statistic:")
        print(f"  Correlation coefficient: {r:.4f}")
        print(f"  Fit quality: {'GOOD' if r > 0.99 else 'SATISFACTORY' if r > 0.98 else 'POOR'}")
        
        return {
            'shapiro': (shapiro_stat, shapiro_p) if n <= 5000 else None,
            'jarque_bera': (jb_stat, jb_p),
            'kolmogorov_smirnov': (ks_stat, ks_p),
            'mean': mean_val,
            'std': std_val,
            'skewness': skew_val,
            'kurtosis': kurt_val,
            'outside_2sigma': outside_2sigma,
            'outside_3sigma': outside_3sigma,
            'qq_correlation': r
        }
    
    def calculate_metrics(self, tvp_corr, dcc_corr, rolling_corr):
        """Calculate metrics"""
        min_len = min(len(tvp_corr), len(dcc_corr), len(rolling_corr))
        
        metrics = {
            'TVP Algorithm': {
                'mean_correlation': np.mean(tvp_corr[:min_len]),
                'std_correlation': np.std(tvp_corr[:min_len]),
                'min_correlation': np.min(tvp_corr[:min_len]),
                'max_correlation': np.max(tvp_corr[:min_len]),
                'positive_percentage': np.mean(tvp_corr[:min_len] > 0) * 100
            },
            'DCC-GARCH': {
                'mean_correlation': np.mean(dcc_corr[:min_len]),
                'std_correlation': np.std(dcc_corr[:min_len]),
                'min_correlation': np.min(dcc_corr[:min_len]),
                'max_correlation': np.max(dcc_corr[:min_len]),
                'positive_percentage': np.mean(dcc_corr[:min_len] > 0) * 100
            },
            'Rolling Correlation': {
                'mean_correlation': np.mean(rolling_corr[:min_len]),
                'std_correlation': np.std(rolling_corr[:min_len]),
                'min_correlation': np.min(rolling_corr[:min_len]),
                'max_correlation': np.max(rolling_corr[:min_len]),
                'positive_percentage': np.mean(rolling_corr[:min_len] > 0) * 100
            }
        }
        
        return metrics
    
    def save_results(self, tvp_results, dcc_results, rolling_corr, output_dir='results/ofz_ruble'):
        """Save results"""
        import os
        import json
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/plots', exist_ok=True)
        os.makedirs(f'{output_dir}/data', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        corr_df = pd.DataFrame({
            'date': self.data['dates'],
            'tvp_correlation': tvp_results['correlation'],
            'tvp_lower_ci': tvp_results['corr_low'],
            'tvp_upper_ci': tvp_results['corr_up'],
            'dcc_correlation': dcc_results['t_copula']['dcc_corr'][:len(self.data['dates'])],
            'rolling_correlation': rolling_corr
        })
        
        corr_df.to_csv(f'{output_dir}/data/correlations_{timestamp}.csv', index=False)
        
        residuals_df = pd.DataFrame({
            'date': self.data['dates'],
            'standardized_residuals': tvp_results['results']['standardized_residuals'],
            'residuals': tvp_results['results']['residuals'],
            'sigma_t': tvp_results['results']['sigma_t_mean']
        })
        residuals_df.to_csv(f'{output_dir}/data/residuals_{timestamp}.csv', index=False)
        
        metrics = self.calculate_metrics(
            tvp_results['correlation'],
            dcc_results['t_copula']['dcc_corr'],
            rolling_corr
        )
        
        with open(f'{output_dir}/data/metrics_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        dcc_params = {
            'classical': {
                'parameters': dcc_results['classical']['params'].tolist(),
                'type': dcc_results['classical']['type']
            },
            't_copula': {
                'parameters': dcc_results['t_copula']['params'].tolist(),
                'type': dcc_results['t_copula']['type']
            }
        }
        
        with open(f'{output_dir}/data/dcc_parameters_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(dcc_params, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_dir}/")
        print(f"  - Correlations: {output_dir}/data/correlations_{timestamp}.csv")
        print(f"  - Residuals: {output_dir}/data/residuals_{timestamp}.csv")
        print(f"  - Metrics: {output_dir}/data/metrics_{timestamp}.json")
        print(f"  - DCC parameters: {output_dir}/data/dcc_parameters_{timestamp}.json")
    
    def run_full_experiment(self):
        """Run full experiment"""
        print("=" * 70)
        print("EXPERIMENT: OFZ AND RUBLE CORRELATION")
        print("=" * 70)
        
        # 1. Load data
        self.load_and_prepare_data()
        
        # 2. Visualize raw data
        fig_raw = DataProcessor.plot_raw_data(
            self.data,
            price1_key='ofz_rates',
            price2_key='ruble_prices',
            growth1_key='ofz_growth',
            growth2_key='ruble_growth',
            label1='OFZ yield (%)',
            label2='Ruble rate (USD/RUB)'
        )
        fig_raw.savefig('ofz_ruble_raw_data.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Run TVP algorithm
        tvp_results = self.run_tvp_algorithm()
        
        # 4. Normality tests
        normality_results = self.test_normality(tvp_results['results']['standardized_residuals'])
        
        # 5. Plot residuals
        print("\nVisualizing standardized residuals...")
        fig_residuals = self.plot_standardized_residuals(tvp_results)
        fig_residuals.savefig('ofz_ruble_standardized_residuals.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 6. Run DCC-GARCH
        dcc_results = self.run_dcc_garch()
        
        # 7. Rolling correlation
        rolling_corr = self.calculate_rolling_correlation()
        
        # 8. Plot results
        fig_results = self.plot_results(tvp_results, dcc_results, rolling_corr)
        fig_results.savefig('ofz_ruble_correlation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 9. Metrics
        metrics = self.calculate_metrics(
            tvp_results['correlation'],
            dcc_results['t_copula']['dcc_corr'],
            rolling_corr
        )
        
        print("\n" + "=" * 70)
        print("CORRELATION METRICS")
        print("=" * 70)
        
        for method, values in metrics.items():
            print(f"\n{method}:")
            print(f"  Mean correlation: {values['mean_correlation']:.4f}")
            print(f"  Standard deviation: {values['std_correlation']:.4f}")
            print(f"  Minimum correlation: {values['min_correlation']:.4f}")
            print(f"  Maximum correlation: {values['max_correlation']:.4f}")
            print(f"  % positive values: {values['positive_percentage']:.1f}%")
        
        # 10. Save results
        self.save_results(tvp_results, dcc_results, rolling_corr)
        
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETED")
        print("=" * 70)
        
        return {
            'data': self.data,
            'tvp_results': tvp_results,
            'normality_results': normality_results,
            'dcc_results': dcc_results,
            'rolling_corr': rolling_corr,
            'metrics': metrics
        }


def main():
    """Main function"""
    config = OFZRubleConfig()
    experiment = OFZRubleExperiment(config)
    results = experiment.run_full_experiment()
    return results


if __name__ == "__main__":
    results = main()
