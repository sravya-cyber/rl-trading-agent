import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN

from env.trading_env import StockTradingEnv
from data.data_loader import download_data, add_features, normalize

def run_backtest(model, env, label="Agent"):
    obs, _ = env.reset()
    net_worths, actions_taken = [env.net_worth], []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        net_worths.append(env.net_worth)
        actions_taken.append(int(action))
        if terminated or truncated:
            break
    return np.array(net_worths), actions_taken

def compute_metrics(net_worths, initial=10000, label="Agent"):
    returns = np.diff(net_worths) / net_worths[:-1]
    total_return = (net_worths[-1] - initial) / initial * 100
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
    peak = np.maximum.accumulate(net_worths)
    drawdown = (net_worths - peak) / peak
    max_drawdown = drawdown.min() * 100

    print(f"\n📊 [{label}] Performance Metrics")
    print(f"  Final Net Worth  : ${net_worths[-1]:,.2f}")
    print(f"  Total Return     : {total_return:.2f}%")
    print(f"  Sharpe Ratio     : {sharpe:.3f}")
    print(f"  Max Drawdown     : {max_drawdown:.2f}%")
    return total_return, sharpe, max_drawdown

def buy_hold_baseline(df, initial=10000):
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close_prices = df["Close"].squeeze().values.flatten()
    start_price = float(close_prices[0])
    shares = initial / start_price
    return shares * close_prices

# ─── Load test data ──────────────────────────────────────────
# ─── Load test data ──────────────────────────────────────────
df_raw  = download_data("AAPL", start="2018-01-01", end="2022-12-31")

# Flatten multi-level columns
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)

df_feat = add_features(df_raw.copy())
df_norm, _ = normalize(df_feat.copy())
split   = int(len(df_norm) * 0.8)
test_df = df_norm.iloc[split:].reset_index(drop=True)
test_df_raw = df_feat.iloc[split:].reset_index(drop=True)

# Flatten test_df_raw columns too
if isinstance(test_df_raw.columns, pd.MultiIndex):
    test_df_raw.columns = test_df_raw.columns.get_level_values(0)

# ─── Load models ─────────────────────────────────────────────
ppo_model = PPO.load("models/best_model")       # best saved by EvalCallback
dqn_model = DQN.load("models/dqn_trading_final")

ppo_env = StockTradingEnv(test_df)
dqn_env = StockTradingEnv(test_df)

ppo_nw, ppo_actions = run_backtest(ppo_model, ppo_env, "PPO")
dqn_nw, dqn_actions = run_backtest(dqn_model, dqn_env, "DQN")
bnh_nw = buy_hold_baseline(test_df_raw)

compute_metrics(ppo_nw, label="PPO")
compute_metrics(dqn_nw, label="DQN")

# ─── Plot ────────────────────────────────────────────────────
plt.figure(figsize=(14, 6))
plt.plot(ppo_nw, label="PPO Agent",        color="royalblue", linewidth=2)
plt.plot(dqn_nw, label="DQN Agent",        color="orange",    linewidth=2)
plt.plot(bnh_nw, label="Buy & Hold",        color="green",     linewidth=2, linestyle="--")
plt.axhline(y=10000, color="red", linestyle=":", label="Initial Capital")
plt.title("Portfolio Performance: RL Agents vs Buy & Hold", fontsize=14)
plt.xlabel("Trading Days")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("backtest/performance_comparison.png", dpi=150)
plt.show()
print("✅ Backtest chart saved!")