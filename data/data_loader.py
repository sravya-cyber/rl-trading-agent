import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

def download_data(ticker="AAPL", start="2018-01-01", end="2023-12-31"):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    
    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.dropna(inplace=True)
    
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}. Check ticker or date range.")
    
    print(f"✅ Downloaded {len(df)} rows for {ticker}")
    return df

def add_features(df):
    close = df["Close"].squeeze()

    # RSI
    df["RSI"] = RSIIndicator(close=close, window=14).rsi()

    # MACD
    macd = MACD(close=close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    # Bollinger Bands
    bb = BollingerBands(close=close)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Pct"] = bb.bollinger_pband()

    # EMA
    df["EMA_20"] = EMAIndicator(close=close, window=20).ema_indicator()

    # Daily returns
    df["Return"] = close.pct_change()

    # Volume change
    df["Volume_Change"] = df["Volume"].pct_change()

    df.dropna(inplace=True)
    return df

def normalize(df):
    from sklearn.preprocessing import MinMaxScaler
    feature_cols = ["Open","High","Low","Close","Volume",
                    "RSI","MACD","MACD_Signal",
                    "BB_Upper","BB_Lower","BB_Pct",
                    "EMA_20","Return","Volume_Change"]
    
    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    scaler = MinMaxScaler(feature_range=(0.01, 1.0))  # avoid zero
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

if __name__ == "__main__":
    df = download_data("AAPL")
    df = add_features(df)
    df, scaler = normalize(df)
    df.to_csv("data/AAPL_features.csv")
    print(df.tail())
    print("✅ Data downloaded and processed!")