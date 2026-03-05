import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.optimize import minimize
from scipy.special import gammaln
from arch import arch_model


class DCCGARCH:
    """Class for estimating DCC-GARCH models to compare with TVP algorithm."""

    @staticmethod
    def vecl(matrix):
        """Vectorization of the lower triangle of a matrix (excluding diagonal)."""
        lower_matrix = np.tril(matrix, k=-1)
        array_with_zero = np.matrix(lower_matrix).A1
        return array_with_zero[array_with_zero != 0]

    @staticmethod
    def dcceq(theta, trdata):
        """DCC model equations (Engle, 2002)."""
        T, N = trdata.shape
        a, b = theta

        Qt = np.zeros((N, N, T))
        Rt = np.zeros((N, N, T))
        veclRt = np.zeros((T, int(N*(N-1)/2)))

        Qt[:, :, 0] = np.cov(trdata.T)
        Rt[:, :, 0] = np.corrcoef(trdata.T)

        for j in range(1, T):
            Qt[:, :, j] = (1 - a - b) * Qt[:, :, 0]
            Qt[:, :, j] += a * np.outer(trdata[j-1, :], trdata[j-1, :])
            Qt[:, :, j] += b * Qt[:, :, j-1]

            diag_Qt = np.sqrt(np.diag(Qt[:, :, j]))
            Rt[:, :, j] = Qt[:, :, j] / np.outer(diag_Qt, diag_Qt)

        for j in range(T):
            veclRt[j, :] = DCCGARCH.vecl(Rt[:, :, j].T)

        return Rt, veclRt

    @staticmethod
    def dcceq_t(theta, trdata, nu):
        """DCC model equations with t-distribution scaling."""
        T, N = trdata.shape
        a, b = theta

        Qt = np.zeros((N, N, T))
        Rt = np.zeros((N, N, T))
        veclRt = np.zeros((T, int(N*(N-1)/2)))

        Qt[:, :, 0] = np.cov(trdata.T)
        Rt[:, :, 0] = np.corrcoef(trdata.T)

        scale_factor = (nu - 2) / nu

        for j in range(1, T):
            Qt[:, :, j] = (1 - a - b) * Qt[:, :, 0]
            Qt[:, :, j] += a * np.outer(trdata[j-1, :], trdata[j-1, :]) * scale_factor
            Qt[:, :, j] += b * Qt[:, :, j-1]

            diag_Qt = np.sqrt(np.diag(Qt[:, :, j]))
            Rt[:, :, j] = Qt[:, :, j] / np.outer(diag_Qt, diag_Qt)

        for j in range(T):
            veclRt[j, :] = DCCGARCH.vecl(Rt[:, :, j].T)

        return Rt, veclRt

    @staticmethod
    def loglike_norm_dcc(theta, trdata):
        """Log-likelihood for classical DCC-GARCH (applied to standardized residuals)."""
        T, N = trdata.shape
        Rt, veclRt = DCCGARCH.dcceq(theta, trdata)

        llf = 0
        for i in range(T):
            Rt_i = Rt[:, :, i]
            inv_Rt_i = np.linalg.inv(Rt_i)
            x_i = trdata[i, :]

            llf += -0.5 * np.log(np.linalg.det(Rt_i))
            llf += -0.5 * x_i @ inv_Rt_i @ x_i

        return -llf

    @staticmethod
    def loglike_t_dcc(theta, trdata):
        """Log-likelihood for DCC-GARCH with t-distribution."""
        T, N = trdata.shape
        a, b, nu = theta[0], theta[1], theta[2]

        Rt, veclRt = DCCGARCH.dcceq_t(theta[:2], trdata, nu)

        llf = 0
        for i in range(T):
            d = N
            Rt_i = Rt[:, :, i]
            inv_Rt_i = np.linalg.inv(Rt_i)
            x_i = trdata[i, :]

            const = gammaln((nu + d)/2) - gammaln(nu/2) - (d/2)*np.log(np.pi*nu)
            llf += const - 0.5*np.log(np.linalg.det(Rt_i))
            llf -= ((nu + d)/2)*np.log(1 + x_i @ inv_Rt_i @ x_i / nu)

        return -llf

    @staticmethod
    def loglike_norm_dcc_copula(theta, udata):
        """Log-likelihood for DCC with normal copula."""
        N, T = udata.shape
        trdata = np.array(norm.ppf(udata).T, ndmin=2)
        return DCCGARCH.loglike_norm_dcc(theta, trdata)

    @staticmethod
    def loglike_t_dcc_copula(theta, udata):
        """Log-likelihood for DCC with t-copula."""
        N, T = udata.shape
        a, b, nu = theta[0], theta[1], theta[2]
        trdata = np.array([t.ppf(udata[i, :], nu) for i in range(N)]).T
        return DCCGARCH.loglike_t_dcc(theta, trdata)

    def fit_classical_dcc(self, rets):
        """Estimate classical DCC-GARCH with normal margins."""
        print("Estimating classical DCC-GARCH...")

        std_resid_list = []
        for column in rets.columns:
            am = arch_model(rets[column], dist='normal')
            res = am.fit(disp='off', show_warning=False)
            std_resid = res.resid / res.conditional_volatility
            std_resid_list.append(std_resid)

        std_resid_matrix = np.column_stack(std_resid_list)

        cons = {'type': 'ineq', 'fun': lambda x: -x[0] - x[1] + 1}
        bnds = ((0, 0.5), (0, 0.9997))

        opt = minimize(
            self.loglike_norm_dcc,
            [0.01, 0.95],
            args=(std_resid_matrix,),
            bounds=bnds,
            constraints=cons,
            method='SLSQP',
            options={'maxiter': 1000, 'ftol': 1e-8}
        )

        Rt, veclRt = self.dcceq(opt.x, std_resid_matrix)
        dcc_corr = veclRt[:, 0]

        results = {
            'dcc_corr': dcc_corr,
            'params': opt.x,
            'type': 'Classical DCC-GARCH'
        }
        return results

    def fit_t_copula_dcc(self, rets):
        """Estimate DCC-GARCH with t-copula."""
        print("Estimating DCC-GARCH with t-copula...")

        udata_list = []
        for column in rets.columns:
            am = arch_model(rets[column], dist='t')
            res = am.fit(disp='off', show_warning=False)

            mu = res.params['mu']
            nu = res.params['nu']
            est_r = rets[column].values - mu
            h = res.conditional_volatility
            std_res = est_r / h
            udata = t.cdf(std_res, nu)
            udata_list.append(udata)

        udata_array = np.array(udata_list)

        cons = {'type': 'ineq', 'fun': lambda x: -x[0] - x[1] + 1}
        bnds = ((0, 0.5), (0, 0.9997), (2.1, 50))

        opt = minimize(
            self.loglike_t_dcc_copula,
            [0.01, 0.95, 5],
            args=(udata_array,),
            bounds=bnds,
            constraints=cons,
            method='SLSQP',
            options={'maxiter': 1000, 'ftol': 1e-8}
        )

        trdata = np.array([t.ppf(udata_array[i, :], opt.x[2]) for i in range(2)]).T
        Rt, veclRt = self.dcceq_t(opt.x[:2], trdata, opt.x[2])
        dcc_corr = veclRt[:, 0]

        results = {
            'dcc_corr': dcc_corr,
            'params': opt.x,
            'type': 'DCC-GARCH (t-copula)'
        }
        return results

    def fit_both_models(self, rets):
        """Estimate both DCC-GARCH models for comparison."""
        results = {}
        results['classical'] = self.fit_classical_dcc(rets)
        results['t_copula'] = self.fit_t_copula_dcc(rets)
        return results
