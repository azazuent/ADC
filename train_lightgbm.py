import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from ta.momentum import RSIIndicator
from ta.trend import MACD, CCIIndicator
from ta.volatility import BollingerBands


def build_dataset(prices, adc, window_size=30, horizon=10):
    df = pd.DataFrame({
        'price': prices,
        'adc': adc
    })

    df['log_ret'] = np.log(df['price'] / df['price'].shift(1)).fillna(0)
    df['ma'] = df['price'].rolling(10).mean()
    df['std'] = df['price'].rolling(10).std()
    df['rsi'] = compute_rsi(df['price'], window=14)

    df['return_1'] = df['price'].pct_change(1)
    df['return_3'] = df['price'].pct_change(3)
    df['return_7'] = df['price'].pct_change(7)
    df['return_14'] = df['price'].pct_change(14)

    # Скользящие средние и разницы
    df['ma_3'] = df['price'].rolling(window=3).mean()
    df['ma_7'] = df['price'].rolling(window=7).mean()
    df['ma_14'] = df['price'].rolling(window=14).mean()
    df['ma_diff_3_7'] = df['ma_3'] - df['ma_7']
    df['ma_diff_7_14'] = df['ma_7'] - df['ma_14']

    # Склон градиента
    df['grad_1'] = df['price'].diff(1)
    df['grad_3'] = df['price'].diff(3)
    df['grad_7'] = df['price'].diff(7)

    # Волатильность
    df['std_3'] = df['price'].rolling(window=3).std()
    df['std_7'] = df['price'].rolling(window=7).std()
    df['std_14'] = df['price'].rolling(window=14).std()

    # RSI + его производные
    rsi = RSIIndicator(df['price'], window=14).rsi()
    df['rsi'] = rsi
    df['rsi_grad'] = rsi.diff(1)
    df['rsi_diff7'] = rsi - rsi.rolling(window=7).mean()

    # MACD
    macd = MACD(df['price'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # CCI (циклический индикатор тренда)
    cci = CCIIndicator(high=df['price'], low=df['price'], close=df['price'], window=20)
    df['cci'] = cci.cci()

    # Bollinger Bands
    bb = BollingerBands(df['price'])
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_bbh'] - df['bb_bbl']

    # Цикличность и сезонность
    df['day_sin'] = np.sin(np.linspace(0, 2 * np.pi, len(df)))
    df['day_cos'] = np.cos(np.linspace(0, 2 * np.pi, len(df)))

    features = ['price', 'log_ret', 'ma', 'std', 'rsi']

    X, y = [], []
    for i in range(window_size, len(df) - horizon):
        x_i = df.iloc[i - window_size:i].values.flatten()
        y_i = df['adc'].iloc[i + horizon]
        if not np.isnan(y_i) and not np.isnan(x_i).any():
            X.append(x_i)
            y.append(y_i)

    return np.array(X), np.array(y)

# --- RSI ---
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

# --- Обучение и метрики ---
def train_lightgbm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = lgb.LGBMRegressor(objective='quantile', alpha=0.5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    acc = accuracy_score(y_test > 0, y_pred > 0)

    print(f"RMSE: {rmse:.4f}")
    print(f"Sign Accuracy: {acc:.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(y_test, label="True ADC")
    plt.plot(y_pred, label="Predicted ADC")
    plt.legend()
    plt.title("ADC Prediction")
    plt.grid(True)
    plt.show()

# --- Демонстрация ---
if __name__ == "__main__":

    prices = pd.read_csv(f"./data/prices/BTC.csv")
    prices = prices["close"].values.flatten()

    adc = pd.read_csv("./data/adc/BTC_2-50.csv")
    adc = adc["adc"].values.flatten()

    X, y = build_dataset(prices, adc, window_size=30, horizon=10)
    train_lightgbm(X, y)