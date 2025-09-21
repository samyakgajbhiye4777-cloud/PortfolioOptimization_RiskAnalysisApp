
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.stMetric {
    background-color: #f0f2f6;
    border: 1px solid #e0e0e0;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.portfolio-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="portfolio-header"><h1>🚀 Advanced Portfolio Optimizer</h1><p>Optimize your Indian & US stock portfolios with advanced analytics and risk management</p></div>', unsafe_allow_html=True)

def validate_and_format_ticker(ticker, market_type="Mixed"):
    """Validate and format ticker symbols"""
    ticker = ticker.upper().strip()

    # Remove any existing suffixes first
    clean_ticker = ticker.replace('.NS', '').replace('.BO', '').replace('.BSE', '')

    # Known US stocks that shouldn't have Indian suffixes
    us_stocks = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ORCL', 
                 'CRM', 'ADBE', 'INTC', 'IBM', 'CSCO', 'AMD', 'QCOM', 'TXN', 'AVGO', 'BABA',
                 'SPY', 'QQQ', 'VOO', 'VTI', 'IVV', 'SCHD', 'XOM', 'CVX', 'JPM', 'BAC', 'WFC',
                 'GS', 'MS', 'C', 'V', 'MA', 'PYPL', 'SQ', 'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK']

    # If it's a known US stock, return as is
    if clean_ticker in us_stocks:
        return clean_ticker

    # If ticker already has .NS or .BO, keep it
    if '.NS' in ticker or '.BO' in ticker or '.BSE' in ticker:
        return ticker

    # If market type is Indian or Mixed, add .NS for unknown tickers
    if market_type in ["Indian Stocks", "Mixed Portfolio"]:
        return f"{clean_ticker}.NS"

    # Otherwise return as is (assuming US stock)
    return clean_ticker

def fetch_stock_data_robust(tickers, period="2y"):
    """Robust data fetching with multiple fallbacks"""
    price_data = pd.DataFrame()
    current_prices = {}
    failed_tickers = []
    successful_tickers = []

    for ticker in tickers:
        try:
            # Try to fetch individual stock data
            stock = yf.Ticker(ticker)

            # Try to get historical data
            hist_data = stock.history(period=period)

            if hist_data.empty:
                st.warning(f"⚠️ No data found for {ticker}")
                failed_tickers.append(ticker)
                continue

            # Use Close price if Adj Close is not available
            if 'Close' in hist_data.columns:
                price_series = hist_data['Close']
            else:
                st.warning(f"⚠️ No Close price data for {ticker}")
                failed_tickers.append(ticker)
                continue

            # Add to price_data DataFrame
            if price_data.empty:
                price_data = pd.DataFrame({ticker: price_series})
            else:
                price_data[ticker] = price_series

            # Get current price
            try:
                info = stock.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', price_series.iloc[-1]))
                if current_price is None or current_price == 0:
                    current_price = price_series.iloc[-1]
                current_prices[ticker] = float(current_price)
            except:
                current_prices[ticker] = float(price_series.iloc[-1])

            successful_tickers.append(ticker)

        except Exception as e:
            st.warning(f"⚠️ Failed to fetch data for {ticker}: {str(e)}")
            failed_tickers.append(ticker)
            continue

    # Remove any NaN values
    if not price_data.empty:
        price_data = price_data.dropna()

    return price_data, current_prices, successful_tickers, failed_tickers

class PortfolioOptimizer:
    def __init__(self, tickers, period="2y", rf=0.055):
        self.tickers = tickers
        self.period = period
        self.rf = rf / 252  # Convert annual risk-free rate to daily
        self.price_data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.successful_tickers = []

    def fetch_data(self):
        """Fetch stock data with robust error handling"""
        try:
            self.price_data, current_prices, self.successful_tickers, failed_tickers = fetch_stock_data_robust(
                self.tickers, self.period
            )

            if self.price_data is None or self.price_data.empty:
                st.error("❌ Failed to fetch data for any of the provided tickers.")
                return None, None

            if failed_tickers:
                st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_tickers)}")

            if len(self.successful_tickers) < 2:
                st.error("❌ Need at least 2 stocks with valid data for portfolio optimization.")
                return None, None

            # Update tickers list to only include successful ones
            self.tickers = self.successful_tickers

            return self.price_data, current_prices

        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            return None, None

    def compute_returns(self):
        """Calculate returns and statistics"""
        if self.price_data is None or self.price_data.empty:
            return None

        self.returns = self.price_data.pct_change().dropna()

        # Handle any remaining NaN or infinite values
        self.returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()

        if self.returns.empty:
            st.error("❌ No valid return data after cleaning.")
            return None

        self.mean_returns = self.returns.mean() * 252  # Annualized
        self.cov_matrix = self.returns.cov() * 252     # Annualized

        return self.returns

    def portfolio_performance(self, weights):
        """Calculate portfolio performance metrics"""
        try:
            port_return = np.dot(weights, self.mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe = (port_return - self.rf * 252) / port_vol if port_vol > 0 else 0
            return port_return, port_vol, sharpe
        except Exception as e:
            st.error(f"Error calculating portfolio performance: {str(e)}")
            return 0, 0, 0

    def neg_sharpe(self, weights):
        return -self.portfolio_performance(weights)[2]

    def portfolio_volatility(self, weights):
        return self.portfolio_performance(weights)[1]

    def maximize_return(self, weights):
        return -self.portfolio_performance(weights)[0]

    def optimize_portfolio(self, objective='max_sharpe'):
        """Optimize portfolio based on objective"""
        try:
            n_assets = len(self.mean_returns)
            bounds = tuple((0, 1) for _ in range(n_assets))
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            initial_guess = np.array([1/n_assets] * n_assets)

            if objective == 'max_sharpe':
                result = minimize(self.neg_sharpe, 
                                x0=initial_guess,
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints,
                                options={'maxiter': 1000})
            elif objective == 'min_vol':
                result = minimize(self.portfolio_volatility,
                                x0=initial_guess,
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints,
                                options={'maxiter': 1000})
            elif objective == 'max_return':
                result = minimize(self.maximize_return,
                                x0=initial_guess,
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints,
                                options={'maxiter': 1000})

            return result.x if result.success else initial_guess

        except Exception as e:
            st.error(f"Error in optimization: {str(e)}")
            n_assets = len(self.mean_returns)
            return np.array([1/n_assets] * n_assets)

    def efficient_frontier(self, num_portfolios=50):
        """Generate efficient frontier with error handling"""
        try:
            n_assets = len(self.mean_returns)
            results = np.zeros((3, num_portfolios))
            weights_array = []

            # Target returns for efficient frontier
            min_ret = self.mean_returns.min()
            max_ret = self.mean_returns.max()
            target_returns = np.linspace(min_ret, max_ret, num_portfolios)

            bounds = tuple((0, 1) for _ in range(n_assets))

            for i, target in enumerate(target_returns):
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                             {'type': 'eq', 'fun': lambda x: np.dot(x, self.mean_returns) - target})

                result = minimize(self.portfolio_volatility,
                                x0=np.array([1/n_assets] * n_assets),
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints,
                                options={'maxiter': 1000})

                if result.success:
                    weights_array.append(result.x)
                    ret, vol, sharpe = self.portfolio_performance(result.x)
                    results[0, i] = ret
                    results[1, i] = vol
                    results[2, i] = sharpe
                else:
                    # Fallback to equal weights
                    equal_weights = np.array([1/n_assets] * n_assets)
                    weights_array.append(equal_weights)
                    ret, vol, sharpe = self.portfolio_performance(equal_weights)
                    results[0, i] = ret
                    results[1, i] = vol
                    results[2, i] = sharpe

            return results, weights_array

        except Exception as e:
            st.error(f"Error generating efficient frontier: {str(e)}")
            return None, None

    def calculate_max_drawdown(self, weights):
        """Calculate maximum drawdown with error handling"""
        try:
            portfolio_returns = (self.returns * weights).sum(axis=1)
            cum_returns = (1 + portfolio_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            return max_drawdown
        except Exception as e:
            st.warning(f"Could not calculate max drawdown: {str(e)}")
            return 0

    def calculate_var_cvar(self, weights, confidence=0.05):
        """Calculate Value at Risk and Conditional Value at Risk"""
        try:
            portfolio_returns = (self.returns * weights).sum(axis=1)
            var = np.percentile(portfolio_returns, confidence * 100)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
            return var, cvar
        except Exception as e:
            st.warning(f"Could not calculate VaR/CVaR: {str(e)}")
            return 0, 0

# Sidebar inputs
with st.sidebar:
    st.header("📊 Portfolio Configuration")

    # Stock selection
    st.subheader("Stock Selection")
    market_type = st.selectbox("Market", ["Indian Stocks", "US Stocks", "Mixed Portfolio"])

    if market_type == "Indian Stocks":
        default_tickers = "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK"
        st.info("💡 Indian stocks will automatically get .NS suffix")
    elif market_type == "US Stocks":
        default_tickers = "AAPL,MSFT,GOOGL,AMZN,TSLA"
        st.info("💡 Use standard US ticker symbols")
    else:
        default_tickers = "RELIANCE,TCS,AAPL,MSFT,GOOGL"
        st.info("💡 Mix Indian and US stocks")

    tickers_input = st.text_area("Stock Tickers (comma-separated)", 
                                value=default_tickers,
                                height=100,
                                help="Enter stock symbols separated by commas. Indian stocks will get .NS automatically.")

    # Time period and risk-free rate
    st.subheader("Analysis Parameters")
    period = st.selectbox("Historical Period", 
                         options=["6mo", "1y", "2y", "3y", "5y"], 
                         index=2)

    rf_rate = st.number_input("Risk-free Rate (%)", 
                             min_value=0.0, max_value=20.0, 
                             value=5.5, step=0.1) / 100

    # Investment amount
    st.subheader("Investment Details")
    investment_amount = st.number_input("Investment Amount", 
                                      min_value=1000, max_value=10000000, 
                                      value=100000, step=1000)

    # Analysis options
    st.subheader("Analysis Options")
    show_efficient_frontier = st.checkbox("Show Efficient Frontier", value=True)
    show_risk_analysis = st.checkbox("Show Risk Analysis", value=True)
    show_correlations = st.checkbox("Show Correlation Matrix", value=True)
    show_individual_charts = st.checkbox("Show Individual Stock Charts", value=False)

    # Run analysis button
    run_analysis = st.button("🚀 Run Portfolio Analysis", type="primary")

# Main analysis
if run_analysis and tickers_input:
    # Parse and validate tickers
    raw_tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]

    if len(raw_tickers) < 2:
        st.error("❌ Please enter at least 2 stock tickers")
        st.stop()

    # Format tickers based on market type
    formatted_tickers = [validate_and_format_ticker(ticker, market_type) for ticker in raw_tickers]

    st.info(f"🔍 Analyzing: {', '.join(formatted_tickers)}")

    # Initialize optimizer
    with st.spinner("🔄 Fetching data and computing metrics..."):
        optimizer = PortfolioOptimizer(formatted_tickers, period=period, rf=rf_rate)
        price_data, current_prices = optimizer.fetch_data()

        if price_data is None:
            st.error("❌ Failed to fetch data. Please check your ticker symbols and try again.")
            st.info("💡 **Troubleshooting Tips:**")
            st.info("- For Indian stocks: Try RELIANCE, TCS, INFY, HDFCBANK")
            st.info("- For US stocks: Try AAPL, MSFT, GOOGL, AMZN")
            st.info("- Make sure ticker symbols are correct and active")
            st.stop()

        returns_data = optimizer.compute_returns()

        if returns_data is None:
            st.error("❌ Failed to compute returns data.")
            st.stop()

    st.success(f"✅ Successfully loaded data for {len(optimizer.successful_tickers)} stocks!")

    # Display current stock prices
    st.header("📈 Current Stock Prices")
    cols = st.columns(min(len(optimizer.successful_tickers), 5))

    for i, ticker in enumerate(optimizer.successful_tickers):
        with cols[i % 5]:
            # Remove suffixes for display
            display_ticker = ticker.replace('.NS', '').replace('.BO', '')
            currency = "₹" if ".NS" in ticker or ".BO" in ticker else "$"
            price = current_prices.get(ticker, 0)
            st.metric(display_ticker, f"{currency}{price:.2f}")

    # Portfolio optimization results
    st.header("🎯 Optimized Portfolios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏆 Max Sharpe Ratio")
        max_sharpe_weights = optimizer.optimize_portfolio('max_sharpe')
        ret, vol, sharpe = optimizer.portfolio_performance(max_sharpe_weights)
        max_dd = optimizer.calculate_max_drawdown(max_sharpe_weights)

        st.metric("Annual Return", f"{ret*100:.2f}%")
        st.metric("Annual Volatility", f"{vol*100:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
        st.metric("Max Drawdown", f"{max_dd*100:.2f}%")

        # Portfolio allocation
        weights_df = pd.DataFrame({
            'Stock': [t.replace('.NS', '').replace('.BO', '') for t in optimizer.successful_tickers],
            'Weight': max_sharpe_weights,
            'Amount': max_sharpe_weights * investment_amount
        })
        st.dataframe(weights_df.style.format({'Weight': '{:.2%}', 'Amount': '{:,.0f}'}))

    with col2:
        st.subheader("🛡️ Minimum Volatility")
        min_vol_weights = optimizer.optimize_portfolio('min_vol')
        ret, vol, sharpe = optimizer.portfolio_performance(min_vol_weights)
        max_dd = optimizer.calculate_max_drawdown(min_vol_weights)

        st.metric("Annual Return", f"{ret*100:.2f}%")
        st.metric("Annual Volatility", f"{vol*100:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
        st.metric("Max Drawdown", f"{max_dd*100:.2f}%")

        weights_df = pd.DataFrame({
            'Stock': [t.replace('.NS', '').replace('.BO', '') for t in optimizer.successful_tickers],
            'Weight': min_vol_weights,
            'Amount': min_vol_weights * investment_amount
        })
        st.dataframe(weights_df.style.format({'Weight': '{:.2%}', 'Amount': '{:,.0f}'}))

    with col3:
        st.subheader("📈 Maximum Return")
        max_ret_weights = optimizer.optimize_portfolio('max_return')
        ret, vol, sharpe = optimizer.portfolio_performance(max_ret_weights)
        max_dd = optimizer.calculate_max_drawdown(max_ret_weights)

        st.metric("Annual Return", f"{ret*100:.2f}%")
        st.metric("Annual Volatility", f"{vol*100:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
        st.metric("Max Drawdown", f"{max_dd*100:.2f}%")

        weights_df = pd.DataFrame({
            'Stock': [t.replace('.NS', '').replace('.BO', '') for t in optimizer.successful_tickers],
            'Weight': max_ret_weights,
            'Amount': max_ret_weights * investment_amount
        })
        st.dataframe(weights_df.style.format({'Weight': '{:.2%}', 'Amount': '{:,.0f}'}))

    # Portfolio allocation charts
    st.header("🥧 Portfolio Allocation Visualization")

    fig = make_subplots(rows=1, cols=3, 
                       specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]],
                       subplot_titles=['Max Sharpe', 'Min Volatility', 'Max Return'])

    display_labels = [t.replace('.NS', '').replace('.BO', '') for t in optimizer.successful_tickers]

    # Max Sharpe pie chart
    fig.add_trace(go.Pie(labels=display_labels,
                        values=max_sharpe_weights,
                        name="Max Sharpe",
                        textinfo='label+percent',
                        textposition='auto'), row=1, col=1)

    # Min Volatility pie chart
    fig.add_trace(go.Pie(labels=display_labels,
                        values=min_vol_weights,
                        name="Min Volatility",
                        textinfo='label+percent',
                        textposition='auto'), row=1, col=2)

    # Max Return pie chart
    fig.add_trace(go.Pie(labels=display_labels,
                        values=max_ret_weights,
                        name="Max Return",
                        textinfo='label+percent',
                        textposition='auto'), row=1, col=3)

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Custom weights section
    st.header("🎛️ Custom Portfolio Weights")

    with st.expander("Set Custom Weights", expanded=False):
        st.write("Adjust the sliders to set custom portfolio weights:")
        custom_weights = []

        cols = st.columns(min(len(optimizer.successful_tickers), 4))
        for i, ticker in enumerate(optimizer.successful_tickers):
            with cols[i % 4]:
                display_ticker = ticker.replace('.NS', '').replace('.BO', '')
                weight = st.slider(f"{display_ticker}", 
                                 min_value=0.0, max_value=1.0, 
                                 value=1.0/len(optimizer.successful_tickers), 
                                 step=0.01,
                                 key=f"weight_{ticker}")
                custom_weights.append(weight)

        # Normalize weights
        total_weight = sum(custom_weights)
        if total_weight > 0:
            custom_weights = [w/total_weight for w in custom_weights]

            # Show custom portfolio performance
            ret, vol, sharpe = optimizer.portfolio_performance(np.array(custom_weights))
            max_dd = optimizer.calculate_max_drawdown(np.array(custom_weights))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Custom Return", f"{ret*100:.2f}%")
            with col2:
                st.metric("Custom Volatility", f"{vol*100:.2f}%")
            with col3:
                st.metric("Custom Sharpe", f"{sharpe:.3f}")
            with col4:
                st.metric("Custom Max DD", f"{max_dd*100:.2f}%")

    # Efficient Frontier
    if show_efficient_frontier:
        st.header("📊 Efficient Frontier Analysis")

        with st.spinner("Computing efficient frontier..."):
            frontier_results, frontier_weights = optimizer.efficient_frontier(30)

        if frontier_results is not None:
            # Plot efficient frontier
            fig = go.Figure()

            # Add efficient frontier
            fig.add_trace(go.Scatter(
                x=frontier_results[1] * 100,
                y=frontier_results[0] * 100,
                mode='lines+markers',
                name='Efficient Frontier',
                line=dict(color='blue', width=3),
                marker=dict(size=4)
            ))

            # Add optimal portfolios
            ret_ms, vol_ms, _ = optimizer.portfolio_performance(max_sharpe_weights)
            ret_mv, vol_mv, _ = optimizer.portfolio_performance(min_vol_weights)
            ret_mr, vol_mr, _ = optimizer.portfolio_performance(max_ret_weights)

            fig.add_trace(go.Scatter(
                x=[vol_ms * 100], y=[ret_ms * 100],
                mode='markers', name='Max Sharpe',
                marker=dict(color='red', size=15, symbol='star')
            ))

            fig.add_trace(go.Scatter(
                x=[vol_mv * 100], y=[ret_mv * 100],
                mode='markers', name='Min Volatility',
                marker=dict(color='green', size=15, symbol='diamond')
            ))

            fig.add_trace(go.Scatter(
                x=[vol_mr * 100], y=[ret_mr * 100],
                mode='markers', name='Max Return',
                marker=dict(color='orange', size=15, symbol='triangle-up')
            ))

            # Add individual stocks
            for i, ticker in enumerate(optimizer.successful_tickers):
                individual_return = optimizer.mean_returns.iloc[i] * 100
                individual_vol = np.sqrt(optimizer.cov_matrix.iloc[i, i]) * 100

                fig.add_trace(go.Scatter(
                    x=[individual_vol], y=[individual_return],
                    mode='markers', 
                    name=ticker.replace('.NS', '').replace('.BO', ''),
                    marker=dict(size=10, symbol='circle')
                ))

            fig.update_layout(
                title="Risk-Return Profile: Efficient Frontier & Optimal Portfolios",
                xaxis_title="Volatility (%)",
                yaxis_title="Expected Return (%)",
                height=600,
                hovermode='closest'
            )

            st.plotly_chart(fig, use_container_width=True)

    # Risk Analysis
    if show_risk_analysis:
        st.header("⚠️ Risk Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Portfolio Risk Metrics")

            # Calculate VaR and CVaR for max sharpe portfolio
            var_95, cvar_95 = optimizer.calculate_var_cvar(max_sharpe_weights, 0.05)
            var_99, cvar_99 = optimizer.calculate_var_cvar(max_sharpe_weights, 0.01)

            risk_metrics = pd.DataFrame({
                'Metric': ['Value at Risk (95%)', 'Value at Risk (99%)', 
                          'Conditional VaR (95%)', 'Conditional VaR (99%)',
                          'Maximum Drawdown', 'Annualized Volatility'],
                'Max Sharpe Portfolio': [f"{var_95*100:.2f}%", f"{var_99*100:.2f}%",
                                       f"{cvar_95*100:.2f}%", f"{cvar_99*100:.2f}%",
                                       f"{optimizer.calculate_max_drawdown(max_sharpe_weights)*100:.2f}%",
                                       f"{optimizer.portfolio_performance(max_sharpe_weights)[1]*100:.2f}%"]
            })

            st.dataframe(risk_metrics, use_container_width=True)

        with col2:
            st.subheader("Individual Stock Volatilities")

            stock_volatilities = pd.DataFrame({
                'Stock': [t.replace('.NS', '').replace('.BO', '') for t in optimizer.successful_tickers],
                'Annual Volatility': [np.sqrt(optimizer.cov_matrix.iloc[i, i]) for i in range(len(optimizer.successful_tickers))]
            })
            stock_volatilities['Annual Volatility %'] = stock_volatilities['Annual Volatility'] * 100

            fig = px.bar(stock_volatilities, x='Stock', y='Annual Volatility %',
                        title="Individual Stock Volatilities")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Correlation Analysis
    if show_correlations:
        st.header("🔗 Correlation Matrix")

        correlation_matrix = optimizer.returns.corr()
        correlation_matrix.index = [t.replace('.NS', '').replace('.BO', '') for t in correlation_matrix.index]
        correlation_matrix.columns = [t.replace('.NS', '').replace('.BO', '') for t in correlation_matrix.columns]

        fig = px.imshow(correlation_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       color_continuous_scale='RdYlBu_r',
                       title="Stock Return Correlations")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Individual stock charts
    if show_individual_charts:
        st.header("📈 Individual Stock Price Charts")

        for ticker in optimizer.successful_tickers:
            display_ticker = ticker.replace('.NS', '').replace('.BO', '')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=price_data.index, 
                y=price_data[ticker],
                mode='lines',
                name=display_ticker,
                line=dict(width=2)
            ))

            fig.update_layout(
                title=f"{display_ticker} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    # Portfolio performance comparison
    st.header("📊 Portfolio Performance Comparison")

    # Calculate cumulative returns for different portfolios
    portfolio_returns_ms = (optimizer.returns * max_sharpe_weights).sum(axis=1)
    portfolio_returns_mv = (optimizer.returns * min_vol_weights).sum(axis=1)
    portfolio_returns_mr = (optimizer.returns * max_ret_weights).sum(axis=1)

    cum_returns_ms = (1 + portfolio_returns_ms).cumprod()
    cum_returns_mv = (1 + portfolio_returns_mv).cumprod()
    cum_returns_mr = (1 + portfolio_returns_mr).cumprod()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cum_returns_ms.index, y=cum_returns_ms,
        mode='lines', name='Max Sharpe',
        line=dict(color='red', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=cum_returns_mv.index, y=cum_returns_mv,
        mode='lines', name='Min Volatility',
        line=dict(color='green', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=cum_returns_mr.index, y=cum_returns_mr,
        mode='lines', name='Max Return',
        line=dict(color='orange', width=3)
    ))

    fig.update_layout(
        title="Cumulative Portfolio Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Export functionality
    st.header("💾 Export Results")

    col1, col2 = st.columns(2)

    with col1:
        # Prepare portfolio weights for export
        export_data = pd.DataFrame({
            'Stock': [t.replace('.NS', '').replace('.BO', '') for t in optimizer.successful_tickers],
            'Ticker': optimizer.successful_tickers,
            'Max_Sharpe_Weight': max_sharpe_weights,
            'Min_Vol_Weight': min_vol_weights,
            'Max_Return_Weight': max_ret_weights,
            'Max_Sharpe_Amount': max_sharpe_weights * investment_amount,
            'Min_Vol_Amount': min_vol_weights * investment_amount,
            'Max_Return_Amount': max_ret_weights * investment_amount,
            'Current_Price': [current_prices.get(ticker, 0) for ticker in optimizer.successful_tickers]
        })

        csv = export_data.to_csv(index=False)
        st.download_button(
            label="📊 Download Portfolio Weights (CSV)",
            data=csv,
            file_name=f"portfolio_optimization_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    with col2:
        # Performance summary
        summary_data = pd.DataFrame({
            'Portfolio': ['Max Sharpe', 'Min Volatility', 'Max Return'],
            'Annual_Return': [
                optimizer.portfolio_performance(max_sharpe_weights)[0],
                optimizer.portfolio_performance(min_vol_weights)[0],
                optimizer.portfolio_performance(max_ret_weights)[0]
            ],
            'Annual_Volatility': [
                optimizer.portfolio_performance(max_sharpe_weights)[1],
                optimizer.portfolio_performance(min_vol_weights)[1],
                optimizer.portfolio_performance(max_ret_weights)[1]
            ],
            'Sharpe_Ratio': [
                optimizer.portfolio_performance(max_sharpe_weights)[2],
                optimizer.portfolio_performance(min_vol_weights)[2],
                optimizer.portfolio_performance(max_ret_weights)[2]
            ]
        })

        summary_csv = summary_data.to_csv(index=False)
        st.download_button(
            label="📈 Download Performance Summary (CSV)",
            data=summary_csv,
            file_name=f"portfolio_performance_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

else:
    # Welcome message with troubleshooting
    st.info("""
    👋 **Welcome to the Advanced Portfolio Optimizer!**

    **📊 Features:**
    - ✅ Multiple optimization strategies (Max Sharpe, Min Volatility, Max Return)
    - ✅ Efficient Frontier visualization
    - ✅ Risk analysis (VaR, CVaR, Max Drawdown)
    - ✅ Correlation analysis
    - ✅ Custom weight assignment
    - ✅ Investment amount calculation
    - ✅ Export functionality

    **🚀 Quick Start:**
    1. **Indian Stocks**: Enter symbols like RELIANCE, TCS, INFY, HDFCBANK
    2. **US Stocks**: Enter symbols like AAPL, MSFT, GOOGL, AMZN
    3. **Mixed**: Combine both markets
    4. Set your investment amount and click "Run Portfolio Analysis"

    **💡 Troubleshooting Tips:**
    - For Indian stocks, .NS suffix is added automatically
    - Make sure ticker symbols are active and correct
    - Try popular stocks if having issues: RELIANCE, TCS, AAPL, MSFT
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 <strong>Advanced Portfolio Optimizer v2.0</strong> | Built by Sanjay Amritraj & Samyak Gajbhiye</p>
    <p>⚠️ <em>Disclaimer: This tool is for educational purposes. Past performance does not guarantee future results.</em></p>
</div>
""", unsafe_allow_html=True)
