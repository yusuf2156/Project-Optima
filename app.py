import streamlit as st
import pandas as pd
from datetime import date
import portfolio_optimizer as po
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def get_optimal_weights(tickers, start, end):
    price_data = po.get_data(tickers, start, end)
    if price_data.empty or len(price_data) < 2:
        return pd.Series(dtype=float)
    mean_returns, cov_matrix = po.calculate_statistics(price_data)
    if mean_returns.isnull().any() or cov_matrix.isnull().values.any():
        return pd.Series(dtype=float)
    optimal_weights_array = po.find_optimal_portfolio(mean_returns, cov_matrix)
    return pd.Series(optimal_weights_array, index=mean_returns.index)

st.set_page_config(layout="wide",
                   page_title="Project Optima",
                   page_icon="💵"

)
st.title("Project Optima: An Interactive Portfolio Optimization Engine")
with st.sidebar:
    st.subheader("User Inputs")
    tickers_input= st.text_input("Enter comma-separated stock tickers",value="AAPL,MSFT,GOOG,AMZN")
    start_date = st.date_input("Start Date",value = date(2021,1,1))
    end_date = st.date_input("End Date",value = date.today())
    simulation_count = st.number_input(
        "Number of Monte Carlo Simulations",
        min_value=1000,
        max_value=5000000,
        value=30000,
        step=1000
    )
    run_button = st.button("Run Optimization")

if run_button:
    cleaned_tickers_input = [ticker.upper().strip() for ticker in tickers_input.split(",")]
    st.header(f"Optimal Portfolio for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    price_data = po.get_data(cleaned_tickers_input,start_date,end_date)
    if not price_data.empty and len(price_data) > 1:
        st.success("Data loaded successfully!")
        mean_returns,cov_matrix = po.calculate_statistics(price_data)
        if mean_returns.isnull().any():
            st.warning("Could not calculate statistics for the selected range.")
        else:
            optimal_weights = po.find_optimal_portfolio(mean_returns,cov_matrix)
            optimal_weights = pd.Series(optimal_weights,index=mean_returns.index)
            optimal_return = np.sum(mean_returns*optimal_weights)
            optimal_variance = np.dot(optimal_weights.T,np.dot(cov_matrix,optimal_weights))
            optimal_volatility = np.sqrt(optimal_variance)
            optimal_sharpe = optimal_return/optimal_volatility
            col1,col2,col3 = st.columns(3)
            with col1:
                st.metric(f"Expected Annual Return",value=f"{optimal_return:.2%}")
            with col2:
                st.metric("Expected Annual Volatility",value=f"{optimal_volatility:.2%}")
            with col3:
                st.metric(f"Sharpe Ratio",value=f"{optimal_sharpe:.2f}")
            st.subheader("Optimal Asset Allocation")
            st.bar_chart(optimal_weights)
            st.header("Historical Allocation Comparison")
            start_date_6m = end_date - pd.DateOffset(months=6)
            start_date_3m = end_date - pd.DateOffset(months=3)
            with st.spinner("Running optimizations for historical periods..."):
                optimal_weights_6m = get_optimal_weights(cleaned_tickers_input, start_date_6m, end_date)
                optimal_weights_3m = get_optimal_weights(cleaned_tickers_input, start_date_3m, end_date)
            comparison_df = pd.DataFrame({
                "User Selected Period": optimal_weights,
            })
            if not optimal_weights_6m.empty:
                comparison_df["Last 6 Months"] = optimal_weights_6m
            if not optimal_weights_3m.empty:
                comparison_df["Last 3 Months"] = optimal_weights_3m
            st.subheader("Optimal Weights Comparison Across Time Periods")
            if len(comparison_df.columns) > 1:
                st.bar_chart(comparison_df)
                st.info("""
                            
                            - **User Selected Period**: The dates you chose in the sidebar.
                            - **Last 6 Months**: The six months leading up to your selected end date.
                            - **Last 3 Months**: The three months leading up to your selected end date.

                            Observe how the 'optimal' weights can change based on recent market performance.
                            """)
            else:
                st.warning(
                    "Could not generate comparison chart.")
            st.header("Efficient Frontier")
            mc_progress_bar = st.progress(0, text="Initializing Monte Carlo simulation...")
            mc_results = po.run_monte_carlo_simulation(
                mean_returns,
                cov_matrix,
                simulation_count,
                progress_bar=mc_progress_bar
            )
            fig = px.scatter(mc_results, x="Risk", y="Return", color="Sharpe Ratio",
                             title="Monte Carlo Simulation: Possible Portfolios")
            fig.add_trace(
                go.Scatter(x=[optimal_volatility], y=[optimal_return], mode="markers",
                           marker=dict(color="red", size=15, symbol="star"), name="Max Sharpe Ratio Portfolio")
            )
            fig.update_layout(xaxis_title="Annualized Volatility (Risk)", yaxis_title="Annualized Return")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(
            "Data could not be loaded for the selected tickers and date range.")

