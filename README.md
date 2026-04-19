# Algorithmic Trading with Reinforcement Learning

> An AI agent that learns to trade stocks using PPO and DQN reinforcement learning algorithms trained on real market data from Yahoo Finance.

---

## Results

| Agent | Final Net Worth | Total Return | Sharpe Ratio | Max Drawdown |
|-------|----------------|--------------|--------------|--------------|
| PPO   | $6,987.58       | -30.12%      | -0.574       | -35.86%      |
| DQN   | $11,081.62      | +10.82%      | 0.551        | -22.46%      |

> Backtested on 1,259 days of AAPL data. DQN outperformed PPO on out-of-sample 2023 data.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Neural network backend |
| Stable-Baselines3 | PPO & DQN training |
| OpenAI Gymnasium | Custom trading environment |
| yfinance | Real stock market data |
| Streamlit | Interactive dashboard |
| TensorBoard | Training visualization |
| Pandas / NumPy | Data processing |
| scikit-learn | Feature normalization |

---

## Project Structure

```
rl_trading_agent/
├── env/
│   └── trading_env.py       # Custom Gym environment
├── data/
│   └── data_loader.py       # Data download & feature engineering
├── agent/
│   └── train.py             # Train PPO & DQN agents
├── backtest/
│   └── evaluate.py          # Backtesting & performance metrics
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── models/                  # Saved model weights
└── logs/                    # TensorBoard training logs
```

---

## How It Works

### 1. Data Pipeline
- Downloads real AAPL stock data from Yahoo Finance (1,259 trading days)
- Engineers 14 technical indicators: RSI, MACD, MACD Signal, Bollinger Bands (Upper/Lower/%), EMA-20, Daily Returns, Volume Change
- Normalizes all features using MinMaxScaler (range 0.01–1.0 to avoid zero division)

### 2. Custom Trading Environment
- Built on OpenAI Gymnasium
- **Observation space:** 14 market features + 3 portfolio features (balance, shares held, net worth)
- **Action space:** 3 discrete actions — Buy (1), Sell (2), Hold (0)
- **Reward:** Change in net worth normalized by initial balance
- **Transaction cost:** 0.1% per trade to simulate real-world friction
- **Holding penalty:** Small negative reward for idle holding to encourage exploration

### 3. RL Agents

**PPO (Proximal Policy Optimization)**
- Policy gradient method — learns by directly improving its trading strategy
- Network: MLP with 2 hidden layers of 256 neurons
- Entropy coefficient: 0.05 (forces exploration)
- Trained for 300,000 timesteps

**DQN (Deep Q-Network)**
- Value-based method — memorizes which actions gave the best historical rewards
- Replay buffer: 100,000 transitions
- Network: MLP with 2 hidden layers of 256 neurons
- Trained for 300,000 timesteps

### 4. Backtesting
- Agents tested on 246 unseen trading days (out-of-sample 2023 data)
- Metrics: Total Return, Sharpe Ratio, Max Drawdown
- Compared against Buy & Hold baseline

### 5. Dashboard
- Built with Streamlit + Plotly
- Shows portfolio value over time
- Displays buy/sell signal markers on price chart
- Supports 4 tickers: AAPL, NVDA, TSLA, SPY

---

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/rl_trading_agent.git
cd rl_trading_agent

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install stable-baselines3[extra] gymnasium shimmy yfinance pandas numpy matplotlib streamlit plotly torch tensorboard ta scikit-learn
```

---

## Run the Project

```bash
# Step 1 — Download and process data
python data/data_loader.py

# Step 2 — Train both agents (~30-45 mins on CPU)
python agent/train.py

# Step 3 — Monitor training (optional)
tensorboard --logdir logs/

# Step 4 — Backtest and evaluate
python backtest/evaluate.py

# Step 5 — Launch dashboard
streamlit run dashboard/app.py
```

---

## Key Concepts

**Why RL for trading?**
Traditional ML predicts prices. RL learns a *strategy* — when to buy, sell, or hold — by interacting with the market environment and maximizing cumulative reward over time.

**Why two agents?**
PPO and DQN use fundamentally different learning approaches. Comparing them shows which strategy generalizes better to unseen market conditions.

**Why did PPO underperform?**
PPO learned a conservative hold-everything strategy (safe but low reward). DQN was more aggressive and correctly identified profitable patterns in the 2023 data, achieving +10.82% return.

**Is this real money?**
No. All trading is simulated with virtual $10,000 capital on historical data. This is a research/educational project.

---

## What the Agent Learns

The agent observes technical indicators and learns patterns like:
- RSI below 30 (oversold) → likely price bounce → **BUY**
- RSI above 70 (overbought) → likely price drop → **SELL**
- MACD bullish crossover → upward momentum → **BUY**
- Price near Bollinger upper band → resistance → **SELL**

It learns these patterns purely through trial and error — no hardcoded rules.

---

## Future Improvements

- Train on more timesteps (500k+) for better convergence
- Add more assets (crypto, forex) for diversification
- Implement portfolio-level multi-asset trading
- Add Alpaca API for live paper trading
- Try more advanced algorithms: SAC, TD3, A3C

---

## Author

Built as a capstone project demonstrating advanced reinforcement learning applied to quantitative finance.
