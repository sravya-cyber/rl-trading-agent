import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env.trading_env import StockTradingEnv
from data.data_loader import download_data, add_features, normalize

# ─── Load Data ───────────────────────────────────────────────
print("📥 Downloading data...")
df_raw = download_data("AAPL", start="2019-01-01", end="2023-01-01")
df_feat = add_features(df_raw.copy())
df_norm, scaler = normalize(df_feat.copy())

# Train/test split (80/20)
split = int(len(df_norm) * 0.8)
train_df = df_norm.iloc[:split].reset_index(drop=True)
test_df  = df_norm.iloc[split:].reset_index(drop=True)
print(f"✅ Train: {len(train_df)} rows | Test: {len(test_df)} rows")

# ─── Create Environments ─────────────────────────────────────
train_env = Monitor(StockTradingEnv(train_df))
eval_env  = Monitor(StockTradingEnv(test_df))

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ─── Callbacks ───────────────────────────────────────────────
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="models/",
    log_path="logs/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
    verbose=1
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="models/checkpoints/",
    name_prefix="ppo_trading"
)

# ─── Train PPO ───────────────────────────────────────────────
print("\n🚀 Training PPO agent...")
ppo_model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="logs/ppo/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,   # ← increase this from 0.01 to 0.05 (forces exploration)
    policy_kwargs=dict(net_arch=[256, 256])
)

ppo_model.learn(
    total_timesteps=300_000,
    callback=[eval_callback, checkpoint_callback],
    tb_log_name="PPO_run"
)
ppo_model.save("models/ppo_trading_final")
print("✅ PPO model saved!")

# ─── Train DQN ───────────────────────────────────────────────
print("\n🚀 Training DQN agent...")
train_env2 = Monitor(StockTradingEnv(train_df))

dqn_model = DQN(
    "MlpPolicy",
    train_env2,
    verbose=1,
    tensorboard_log="logs/dqn/",
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[256, 256])
)

dqn_model.learn(
    total_timesteps=200_000,
    tb_log_name="DQN_run"
)
dqn_model.save("models/dqn_trading_final")
print("✅ DQN model saved!")
print("\n🎉 Training complete! Run: tensorboard --logdir logs/")