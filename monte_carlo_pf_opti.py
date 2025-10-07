"""
Monte Carlo Portfolio Simulation with Real Market Data (yfinance)
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance, with fallback Stooq
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    print("yfinance imported successfully - Real market data available")
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance not available - Will use Pandas datareader")
    print("Install with: pip install yfinance")

# Set random seed for reproducibility
np.random.seed(42)

class MonteCarloSimulator:
    """Portfolio simulation class with real-market data via yfinance OR Pandas Datareader"""

    def __init__(self, tickers, start_date, end_date, initial_investment=1000000, use_real_data=True):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_investment = initial_investment
        self.use_real_data = use_real_data and YFINANCE_AVAILABLE
        self.stock_data = None
        self.daily_returns = None
        self.mc_results = None
        self.optimal_portfolio = None
        self.portfolio_tracking = None
        self.data_source = None

        print("MonteCarloSimulator Initialized")
        print(f"Tickers: {self.tickers}")
        print(f"Period: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")
        print(f"Initial Investment: ${initial_investment:,.0f}")
        print(f"Real Data Mode: {'Enabled' if self.use_real_data else 'Disabled (simulation)'}")

    # =============== Data Acquisition ===============
    def _download_real_data(self):
        if not YFINANCE_AVAILABLE:
            return None
        print("Downloading real market data from Yahoo Finance...")
        data = yf.download(
            tickers=self.tickers,
            start=self.start_date.strftime('%Y-%m-%d'),
            end=self.end_date.strftime('%Y-%m-%d'),
            progress=False,
            group_by='ticker'
        )
        if len(self.tickers) == 1:
            data.columns = pd.MultiIndex.from_product([self.tickers, data.columns])
        adj = {}
        for t in self.tickers:
            try:
                adj[t] = data[t]['Adj Close']
            except KeyError:
                adj[t] = data[t]['Close']
        df = pd.DataFrame(adj).dropna()
        if df.empty:
            return None
        self.data_source = 'Yahoo Finance (yfinance)'
        print(f"Downloaded {len(df)} trading days of data")
        return df

    def _real_data_stooq(self, initial_prices=None):
        """
        Fallback data loader using external APIs instead of simulation.
        Tries Stooq via pandas-datareader.
        """
    # 1) Stooq via pandas-datareader
    
        from pandas_datareader import data as pdr
        print("Fetching fallback market data from Stooq (pandas-datareader)...")
        frames = []
        for t in self.tickers:
            # Stooq returns newest->oldest; sort ascending and use Close
            df = pdr.DataReader(t, "stooq",
                                self.start_date, self.end_date)[["Close"]]
            df = df.sort_index().rename(columns={"Close": t})
            frames.append(df)
        real_data = pd.concat(frames, axis=1).dropna()
        real_data.index.name = "Date"
        if real_data.empty:
            raise ValueError("Empty data from Stooq")
        self.data_source = "Stooq (pandas-datareader)"
        return real_data
    

    
    def get_stock_data(self):
        if self.use_real_data:
            real = self._download_real_data()
            if real is not None and len(real) > 100:
                self.stock_data = real
                return self.stock_data
            print("Yfinance data unavailable/insufficient. Falling back to Pandas Datareader - Stooq.")
        self.stock_data = self._real_data_stooq()
        return self.stock_data

    # =============== Analytics ===============
    def calculate_returns(self):
        if self.stock_data is None:
            raise ValueError('Call get_stock_data() first')
        self.daily_returns = self.stock_data.pct_change().dropna()
        return self.daily_returns

    def monte_carlo(self, num_portfolios=10000, risk_free_rate=0.02):
        if self.daily_returns is None:
            raise ValueError('Call calculate_returns() first')
        n = len(self.daily_returns.columns)
        cov_a = self.daily_returns.cov() * 252
        ret_a = self.daily_returns.mean() * 252
        results = np.zeros((4+n, num_portfolios))
        for i in range(num_portfolios):
            w = np.random.random(n)
            w /= w.sum()
            r = float((ret_a * w).sum())
            v = float(np.sqrt(w.T @ cov_a @ w))
            s = (r - risk_free_rate)/v if v > 0 else 0.0
            results[0,i] = r; results[1,i]=v; results[2,i]=s; results[3,i]=self.initial_investment
            results[4:,i] = w
        cols = ['Portfolio_Return','Volatility','Sharpe_Ratio','Portfolio_Value'] + [f'Weight_{t}' for t in self.daily_returns.columns]
        self.mc_results = pd.DataFrame(results.T, columns=cols)
        self.optimal_portfolio = self.mc_results.iloc[self.mc_results['Sharpe_Ratio'].idxmax()]
        return self.mc_results

    def simulate_performance(self):
        if self.optimal_portfolio is None:
            raise ValueError('Run monte_carlo() first')
        weights = {t: self.optimal_portfolio[f'Weight_{t}'] for t in self.tickers}
        start_prices = self.stock_data.iloc[0]
        shares = {t: (self.initial_investment*weights[t])/start_prices[t] for t in self.tickers}
        values = (self.stock_data * pd.Series(shares)).sum(axis=1)
        self.portfolio_tracking = pd.DataFrame({'Date': values.index, 'Portfolio_Value': values.values})
        self.portfolio_tracking['Daily_Return'] = self.portfolio_tracking['Portfolio_Value'].pct_change()
        self.portfolio_tracking['Cumulative_Return'] = (self.portfolio_tracking['Portfolio_Value']/self.initial_investment - 1)*100
        return self.portfolio_tracking

    # =============== Plotting ===============
    def plot_correlation_heatmap(self):
        if self.daily_returns is None:
            raise ValueError('Call calculate_returns() first')
        corr = self.daily_returns.corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=True, cmap='mako', vmin=-1, vmax=1, fmt='.2f', square=True)
        plt.title('Portfolio Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    def plot_efficient_frontier(self):
        if self.mc_results is None:
            raise ValueError('Run monte_carlo() first')
        plt.figure(figsize=(10,7))
        sc = plt.scatter(self.mc_results['Volatility']*100, 
                         self.mc_results['Portfolio_Return']*100,
                         c=self.mc_results['Sharpe_Ratio'], cmap='plasma', s=12, alpha=0.7)
        plt.colorbar(sc, label='Sharpe Ratio')
        # highlight optimal
        opt = self.optimal_portfolio
        plt.scatter(opt['Volatility']*100, opt['Portfolio_Return']*100, c='black', s=80, marker='X', label='Max Sharpe')
        plt.xlabel('Volatility (%)')
        plt.ylabel('Annual Return (%)')
        plt.title('Monte Carlo Efficient Frontier')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_portfolio_value(self):
        if self.portfolio_tracking is None:
            raise ValueError('Run simulate_performance() first')
        plt.figure(figsize=(12,6))
        plt.plot(self.portfolio_tracking['Date'], self.portfolio_tracking['Portfolio_Value']/1e6, color='#d1495b')
        plt.ylabel('Portfolio Value (Millions $)')
        plt.title('Total Portfolio Value [$]')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def run_monte_carlo(tickers=['AMZN','AAPL','GOOGL','MSFT','META','TSLA','JPM','JNJ','PG', 'IBM', 'USAR'], use_real_data=True, years_back=3, num_simulations=10000):
    if tickers is None:
        tickers = ['AMZN','AAPL','GOOGL','MSFT','META','TSLA','JPM','JNJ','PG','CAT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years_back)
    sim = MonteCarloSimulator(tickers, start_date, end_date, 1_000_000, use_real_data)
    # Pipeline
    sim.get_stock_data()
    sim.calculate_returns()
    sim.monte_carlo(num_portfolios=num_simulations)
    sim.simulate_performance()
    # Plots
    sim.plot_correlation_heatmap()
    sim.plot_efficient_frontier()
    sim.plot_portfolio_value()
    # Console summary
    opt = sim.optimal_portfolio
    print("Optimal portfolio (Max Sharpe):")
    print(f"Sharpe: {opt['Sharpe_Ratio']:.4f} | Return: {opt['Portfolio_Return']:.2%} | Volatility: {opt['Volatility']:.2%}")
    return sim


if __name__ == '__main__':
    run_monte_carlo(use_real_data=True, years_back=3, num_simulations=10000)
