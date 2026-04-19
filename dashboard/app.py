import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.trading_env import StockTradingEnv
from data.data_loader import download_data, add_features, normalize

st.set_page_config(page_title="RL Trading Agent", layout="wide", page_icon="📈")
st.title("📈 RL Trading Agent — Capstone Dashboard")

# ─── Sidebar ─────────────────────────────────────────────────
st.sidebar.header("Settings")
ticker   = st.sidebar.text_input("Stock Ticker", "AAPL")
start    = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end      = st.sidebar.date_input("End Date",   pd.to_datetime("2023-01-01"))
capital  = st.sidebar.number_input("Initial Capital ($)", value=10000)

if st.sidebar.button("🚀 Run Agent"):
    with st.spinner("Loading data and running agent..."):
        # Data
        df_raw  = download_data(ticker, str(start), str(end))
        df_feat = add_features(df_raw.copy())
        df_norm, _ = normalize(df_feat.copy())
        test_env = StockTradingEnv(df_norm, initial_balance=capital)

        # Load model
        model = PPO.load("models/best_model")
        obs, _ = test_env.reset()

        portfolio, prices, buys, sells = [capital], [], [], []

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = test_env.step(int(action))
            portfolio.append(test_env.net_worth)
            prices.append(float(df_norm["Close"].iloc[min(test_env.current_step, len(df_norm)-1)]))

            if action == 1:
                buys.append(test_env.current_step - 1)
            elif action == 2:
                sells.append(test_env.current_step - 1)

            if terminated or truncated:
                break

    # ─── Metrics ─────────────────────────────────────────────
    final_worth = portfolio[-1]
    total_return = (final_worth - capital) / capital * 100
    rets = np.diff(portfolio) / np.array(portfolio[:-1])
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252)
    peak = np.maximum.accumulate(portfolio)
    max_dd = ((np.array(portfolio) - peak) / peak).min() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Portfolio", f"${final_worth:,.2f}", f"{total_return:+.2f}%")
    col2.metric("Total Return",    f"{total_return:.2f}%")
    col3.metric("Sharpe Ratio",    f"{sharpe:.3f}")
    col4.metric("Max Drawdown",    f"{max_dd:.2f}%")

    # ─── Portfolio Chart ─────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=portfolio, name="Portfolio Value",
                             line=dict(color="royalblue", width=2)))
    fig.update_layout(title="Portfolio Value Over Time", xaxis_title="Step",
                      yaxis_title="Value ($)", height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # ─── Buy/Sell Signals ────────────────────────────────────
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=prices, name="Price", line=dict(color="gray")))
    if buys:
        fig2.add_trace(go.Scatter(x=buys, y=[prices[i] for i in buys if i < len(prices)],
                                  mode="markers", name="Buy",
                                  marker=dict(color="green", symbol="triangle-up", size=10)))
    if sells:
        fig2.add_trace(go.Scatter(x=sells, y=[prices[i] for i in sells if i < len(prices)],
                                  mode="markers", name="Sell",
                                  marker=dict(color="red", symbol="triangle-down", size=10)))
    fig2.update_layout(title="Buy / Sell Signals", height=400, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.success("✅ Agent simulation complete!")

else:
    st.info("👈 Configure settings and click **Run Agent** to start.")