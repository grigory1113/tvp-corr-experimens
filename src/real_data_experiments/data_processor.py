import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t, norm
import matplotlib.pyplot as plt

class DataProcessor:
    """Обработка данных для нефти и рубля"""
    
    @staticmethod
    def load_and_prepare_data(oil_path='data/WCOILBRENTEU.csv',
                            ruble_path='data/RC_F01_07_1992_T10_06_2025.xlsx',
                            frequency='Q'):
        """
        Загрузка и подготовка данных

        Параметры:
        -----------
        frequency : str
            Частота агрегации: 'W' (неделя), 'Q' (квартал), 'M' (месяц)
        """
        # Загрузка данных
        weekly_oil = pd.read_csv(oil_path)
        daily_usd = pd.read_excel(ruble_path)
        
        # Преобразование дат
        weekly_oil['observation_date'] = pd.to_datetime(weekly_oil['observation_date'])
        daily_usd['data'] = pd.to_datetime(daily_usd['data'])
        
        # Агрегация по частоте
        if frequency == 'Q':
            # Квартальные данные
            quart_avg = (daily_usd.groupby(pd.Grouper(key='data', freq='Q'))['curs']
                         .mean()
                         .reset_index())
            quart_oil = (weekly_oil.groupby(pd.Grouper(key='observation_date', freq='Q'))['WCOILBRENTEU']
                         .mean()
                         .reset_index())
            
            quart_oil['observation_date'] = pd.to_datetime(quart_oil['observation_date'])
            X = quart_avg[quart_avg['data'] <= max(quart_oil['observation_date'])]
            Y = quart_oil[quart_oil['observation_date'] >= min(quart_avg['data'])]
            
        elif frequency == 'M':
            # Месячные данные
            monthly_avg = (daily_usd.groupby(pd.Grouper(key='data', freq='M'))['curs']
                          .mean()
                          .reset_index())
            monthly_oil = (weekly_oil.groupby(pd.Grouper(key='observation_date', freq='M'))['WCOILBRENTEU']
                          .mean()
                          .reset_index())
            
            monthly_oil['observation_date'] = pd.to_datetime(monthly_oil['observation_date'])
            X = monthly_avg[monthly_avg['data'] <= max(monthly_oil['observation_date'])]
            Y = monthly_oil[monthly_oil['observation_date'] >= min(monthly_avg['data'])]
            
        elif frequency == 'W':
            # Недельные данные
            weekly_avg = (daily_usd.groupby(pd.Grouper(key='data', freq='W'))['curs']
                         .mean()
                         .reset_index())
            weekly_avg['data'] = weekly_avg['data'] + pd.Timedelta(days=1)
            
            X = weekly_avg[weekly_avg['data'] <= max(weekly_oil['observation_date'])]
            Y = weekly_oil[weekly_oil['observation_date'] >= min(weekly_avg['data'])]
        
        # Расчёт процентных изменений
        X['price_growth'] = X['curs'].pct_change() * 100
        Y['price_growth'] = Y['WCOILBRENTEU'].pct_change() * 100
        
        # Удаление NaN
        X = X.dropna().reset_index(drop=True)
        Y = Y.dropna().reset_index(drop=True)
        
        # Выравнивание длин
        min_len = min(len(X), len(Y))
        X = X.head(min_len)
        Y = Y.head(min_len)
        
        # Проверка совпадения длин
        if len(X) != len(Y):
            raise ValueError(f"Длины не совпадают: X={len(X)}, Y={len(Y)}")
        
        # Подготовка данных для модели
        n = len(X)
        z = Y['price_growth'].to_numpy().reshape(n, 1)
        x_raw = X['price_growth'].to_numpy().reshape(n, 1)
        x = np.concatenate((np.ones([n, 1]), x_raw), axis=1)
        
        # Метки времени
        dates = X['data'].values
        
        return {
            'z': z,
            'x': x,
            'x_raw': x_raw,
            'dates': dates,
            'oil_prices': Y['WCOILBRENTEU'].values,
            'ruble_prices': X['curs'].values,
            'oil_growth': Y['price_growth'].values,
            'ruble_growth': X['price_growth'].values,
            'n': n
        }

    
    @staticmethod
    def load_ofz_and_ruble_data(ofz_path='data/ofz_data.csv',
                                  ruble_path='data/RC_F01_07_1992_T10_06_2025.xlsx',
                                  frequency='Q'):
        """
        Загрузка и подготовка данных по ОФЗ и рублю

        Параметры:
        -----------
        frequency : str
            Частота агрегации: 'Q' (квартал), 'M' (месяц), 'W' (неделя)
        """
        # Загрузка данных ОФЗ
        ofz_df = pd.read_csv(ofz_path)
        ofz_df['Date'] = pd.to_datetime(ofz_df['Date'], format='%d.%m.%Y')
        ofz_df = ofz_df.rename(columns={'Value': 'ofz_rate'})
        
        # Загрузка данных рубля
        daily_usd = pd.read_excel(ruble_path)
        daily_usd['data'] = pd.to_datetime(daily_usd['data'])
        
        # Агрегация по частоте
        if frequency == 'Q':
            ofz_quart = (ofz_df.groupby(pd.Grouper(key='Date', freq='Q'))['ofz_rate']
                          .mean()
                          .reset_index())
            ruble_quart = (daily_usd.groupby(pd.Grouper(key='data', freq='Q'))['curs']
                           .mean()
                           .reset_index())
            
            ofz_quart = ofz_quart[ofz_quart['Date'] <= ruble_quart['data'].max()]
            ruble_quart = ruble_quart[ruble_quart['data'] >= ofz_quart['Date'].min()]
            
            X = ruble_quart.rename(columns={'data': 'date', 'curs': 'price'})
            Y = ofz_quart.rename(columns={'Date': 'date', 'ofz_rate': 'rate'})
            
        elif frequency == 'M':
            ofz_month = (ofz_df.groupby(pd.Grouper(key='Date', freq='M'))['ofz_rate']
                          .mean()
                          .reset_index())
            ruble_month = (daily_usd.groupby(pd.Grouper(key='data', freq='M'))['curs']
                           .mean()
                           .reset_index())
            
            ofz_month = ofz_month[ofz_month['Date'] <= ruble_month['data'].max()]
            ruble_month = ruble_month[ruble_month['data'] >= ofz_month['Date'].min()]
            
            X = ruble_month.rename(columns={'data': 'date', 'curs': 'price'})
            Y = ofz_month.rename(columns={'Date': 'date', 'ofz_rate': 'rate'})
            
        elif frequency == 'W':
            ofz_week = (ofz_df.groupby(pd.Grouper(key='Date', freq='W'))['ofz_rate']
                         .mean()
                         .reset_index())
            ruble_week = (daily_usd.groupby(pd.Grouper(key='data', freq='W'))['curs']
                          .mean()
                          .reset_index())
            ruble_week['data'] = ruble_week['data'] + pd.Timedelta(days=1)
            
            ofz_week = ofz_week[ofz_week['Date'] <= ruble_week['data'].max()]
            ruble_week = ruble_week[ruble_week['data'] >= ofz_week['Date'].min()]
            
            X = ruble_week.rename(columns={'data': 'date', 'curs': 'price'})
            Y = ofz_week.rename(columns={'Date': 'date', 'ofz_rate': 'rate'})
        
        # Расчёт процентных изменений
        X['growth'] = X['price'].pct_change() * 100
        Y['growth'] = Y['rate'].pct_change() * 100
        
        # Удаление NaN
        X = X.dropna().reset_index(drop=True)
        Y = Y.dropna().reset_index(drop=True)
        
        # Выравнивание длин
        min_len = min(len(X), len(Y))
        X = X.head(min_len)
        Y = Y.head(min_len)
        
        if len(X) != len(Y):
            raise ValueError(f"Длины не совпадают: X={len(X)}, Y={len(Y)}")
        
        # Подготовка данных для модели
        n = len(X)
        z = Y['growth'].to_numpy().reshape(n, 1)          # рост ОФЗ (зависимая)
        x_raw = X['growth'].to_numpy().reshape(n, 1)      # рост рубля (регрессор)
        x = np.concatenate((np.ones([n, 1]), x_raw), axis=1)  # добавление константы
        
        dates = X['date'].values
        
        return {
            'z': z,
            'x': x,
            'x_raw': x_raw,
            'dates': dates,
            'ofz_rates': Y['rate'].values,
            'ruble_prices': X['price'].values,
            'ofz_growth': Y['growth'].values,
            'ruble_growth': X['growth'].values,
            'n': n
        }
    
    @staticmethod
    def estimate_x_variance(x_raw, method='garch', model_params=None):
        """
        Оценка дисперсии ряда x

        Параметры:
        -----------
        method : str
            'garch' - модель GARCH
            'rolling' - скользящее окно
            'ewma' - экспоненциально взвешенное скользящее среднее
        model_params : dict
            Параметры модели

        Возвращает:
        --------
        x_variance : array
            Оценка дисперсии x
        model_results : dict
            Результаты моделирования
        """
        if method == 'garch':
            if model_params is None:
                model_params = {'p': 1, 'q': 0, 'mean': 'constant', 'dist': 'normal'}
            
            model = arch_model(x_raw.flatten(), 
                              vol='Garch', 
                              p=model_params.get('p', 1), 
                              q=model_params.get('q', 0), 
                              mean='constant', 
                              dist=model_params.get('dist', 'normal'))
            
            res = model.fit(disp='off')
            x_variance = res.conditional_volatility ** 2
            
            return x_variance, {'model': res}
            
        elif method == 'rolling':
            if model_params is None:
                model_params = {'window': 20}
            
            window = model_params.get('window', 20)
            x_variance = pd.Series(x_raw.flatten()).rolling(window=window).var().values
            
            if len(x_variance) > window:
                x_variance[:window] = x_variance[window]
            else:
                x_variance[:] = np.var(x_raw.flatten())
            
            return x_variance, {}
            
        elif method == 'ewma':
            if model_params is None:
                model_params = {'alpha': 0.06}
            
            alpha = model_params.get('alpha', 0.06)
            squared_returns = x_raw.flatten() ** 2
            n = len(squared_returns)
            ewma_variance = np.zeros(n)
            ewma_variance[0] = np.var(x_raw[:min(10, n)])
            
            for i in range(1, n):
                ewma_variance[i] = (1 - alpha) * ewma_variance[i-1] + alpha * squared_returns[i]
            
            return ewma_variance, {}
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    
    @staticmethod
    def estimate_garch_t_model(returns):
        """
        Оценка GARCH-t модели

        Возвращает:
        --------
        udata : array
            Данные, преобразованные в равномерное распределение
        model : fitted model
            Оценённая GARCH-t модель
        """
        am = arch_model(returns, dist='t', vol='Garch', p=1, q=1)
        res = am.fit(disp='off', show_warning=False)
        
        mu = res.params['mu']
        nu = res.params['nu']
        est_r = returns - mu
        h = res.conditional_volatility
        std_res = est_r / h
        udata = t.cdf(std_res, nu)
        
        return udata, res
    
    @staticmethod
    def prepare_data_for_dcc_t_copula(data):
        """
        Подготовка данных для DCC с t-копулой

        Параметры:
        -----------
        data : dict
            Данные из load_and_prepare_data

        Возвращает:
        --------
        udata_list : list
            Список преобразованных рядов
        model_parameters : dict
            Параметры GARCH-t моделей
        """
        print("Оценка маржинальных GARCH-t моделей...")
        
        udata_list = []
        model_parameters = {}
        
        # Для нефти
        oil_udata, oil_model = DataProcessor.estimate_garch_t_model(
            data['oil_growth']
        )
        udata_list.append(oil_udata)
        model_parameters['oil'] = oil_model
        
        # Для рубля
        ruble_udata, ruble_model = DataProcessor.estimate_garch_t_model(
            data['ruble_growth']
        )
        udata_list.append(ruble_udata)
        model_parameters['ruble'] = ruble_model
        
        # Преобразование в массив (2 x n)
        udata_array = np.array(udata_list)
        
        print(f"Параметры GARCH-t для нефти: mu={oil_model.params['mu']:.4f}, "
              f"omega={oil_model.params['omega']:.6f}, alpha={oil_model.params['alpha[1]']:.4f}, "
              f"beta={oil_model.params['beta[1]']:.4f}, nu={oil_model.params['nu']:.2f}")
        
        print(f"Параметры GARCH-t для рубля: mu={ruble_model.params['mu']:.4f}, "
              f"omega={ruble_model.params['omega']:.6f}, alpha={ruble_model.params['alpha[1]']:.4f}, "
              f"beta={ruble_model.params['beta[1]']:.4f}, nu={ruble_model.params['nu']:.2f}")
        
        return udata_array, model_parameters
    
    @staticmethod
    def calculate_rolling_correlation(z, x_raw, window=30):
        """
        Расчёт скользящей корреляции

        Параметры:
        -----------
        window : int
            Размер окна
        """
        n = len(z)
        correlations = []
        
        for i in range(n - window):
            corr = np.corrcoef(z[i:i+window].flatten(), 
                              x_raw[i:i+window].flatten())[0, 1]
            correlations.append(corr)
        
        padding = window // 2
        correlations = [correlations[0]] * padding + correlations + [correlations[-1]] * padding
        
        return np.array(correlations[:n])
    
    @staticmethod
    def plot_raw_data(data, price1_key='oil_prices', price2_key='ruble_prices',
                      growth1_key='oil_growth', growth2_key='ruble_growth',
                      label1='Цена нефти ($)', label2='Курс рубля (USD/RUB)'):
        """
        Визуализация исходных данных с настраиваемыми ключами и подписями
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Цены / доходности
        axes[0].plot(data['dates'], data[price1_key], label=label1, color='blue', alpha=0.7)
        axes[0].set_ylabel('Значение')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Динамика цен / доходности')
        
        ax2 = axes[0].twinx()
        ax2.plot(data['dates'], data[price2_key], label=label2, color='red', alpha=0.7)
        ax2.set_ylabel('Значение')
        ax2.legend(loc='upper right')
        
        # Процентные изменения
        axes[1].plot(data['dates'], data[growth1_key], label=f'{label1.split()[0]}, %', color='blue', alpha=0.7)
        axes[1].plot(data['dates'], data[growth2_key], label=f'{label2.split()[0]}, %', color='red', alpha=0.7)
        axes[1].set_ylabel('Изменение, %')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Процентные изменения')
        
        # Дисперсия x (рубля)
        if 'x_variance' in data:
            axes[2].plot(data['dates'], data['x_variance'], label='Дисперсия изменений рубля', color='green', alpha=0.7)
            axes[2].set_ylabel('Дисперсия')
            axes[2].set_xlabel('Дата')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].set_title('Оценка дисперсии изменений курса рубля')
        
        plt.tight_layout()
        return fig
