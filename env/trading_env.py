import gymnasium as gym
import numpy as np
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    """Custom Gymnasium environment for stock trading."""
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, initial_balance=10000, transaction_cost=0.001):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # Feature columns (exclude Date if present)
        self.feature_cols = ["Open","High","Low","Close","Volume",
                             "RSI","MACD","MACD_Signal",
                             "BB_Upper","BB_Lower","BB_Pct",
                             "EMA_20","Return","Volume_Change"]

        n_features = len(self.feature_cols)

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observation: market features + [balance, shares_held, net_worth]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features + 3,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.total_reward = 0
        self.trade_history = []
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step][self.feature_cols]
        if hasattr(row, 'values'):
            row = row.values.flatten()
        extra = np.array([
            self.balance / self.initial_balance,
            self.shares_held / 100.0,
            self.net_worth / self.initial_balance
        ])
        return np.concatenate([row, extra]).astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["Close"]
        if hasattr(current_price, 'iloc'):
            current_price = float(current_price.iloc[0])
        else:
            current_price = float(current_price)
    
        current_price = max(current_price, 1e-6)  # zero guard
        self.prev_net_worth = self.net_worth

        # Execute action
        if action == 1:   # BUY
            shares_to_buy = int(self.balance // (current_price * (1 + self.transaction_cost)))
            cost = shares_to_buy * current_price * (1 + self.transaction_cost)
            self.balance -= cost
            self.shares_held += shares_to_buy
            if shares_to_buy > 0:
                self.trade_history.append(("BUY", self.current_step, current_price))

        elif action == 2:  # SELL
            if self.shares_held > 0:
                revenue = self.shares_held * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.trade_history.append(("SELL", self.current_step, current_price))
                self.shares_held = 0

        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price

        # Reward = change in net worth (Sharpe-inspired)
        reward = (self.net_worth - self.prev_net_worth) / self.initial_balance
        reward = float(np.clip(reward, -1, 1))  # clip for stability

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        reward = (self.net_worth - self.prev_net_worth) / self.initial_balance
        if action == 0 and self.shares_held == 0:
            reward -= 0.0001  # small penalty for doing nothing

        reward = float(np.clip(reward, -1, 1))
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        profit = self.net_worth - self.initial_balance
        print(f"Step: {self.current_step} | Net Worth: ${self.net_worth:.2f} | Profit: ${profit:.2f}")