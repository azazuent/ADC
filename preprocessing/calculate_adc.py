import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

w_min = 2
w_max = 50
alpha_func = lambda w: 1.0 / w
smooth_func = lambda x: np.tanh(3 * x)
# smooth_func=lambda x: x,
epsilon = 1e-8
use_log = True

import numpy as np

def aggregated_directional_change(
    prices: np.ndarray,
    w_min: int = 1,
    w_max: int = 50,
    alpha_func=None,
    smooth_func=None,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Вычисляет Aggregated Directional Change с симметричными окнами:
    (x_{t+w} - x_{t-w}) / (x_t + ε)
    """
    prices = prices.flatten()
    n = len(prices)

    if alpha_func is None:
        alpha_func = lambda w: 1.0 / w
    if smooth_func is None:
        smooth_func = lambda x: x

    ws = np.arange(w_min, w_max + 1)
    weights = np.array([alpha_func(w) for w in ws])

    adc = np.full(n, np.nan)

    for i, w in enumerate(ws):
        left = np.roll(prices, w)
        right = np.roll(prices, -1 * w)
        center = prices

        # Вырезаем края, чтобы не было артефактов от np.roll
        valid_range = np.arange(w, n - w)

        delta = (right[valid_range] - left[valid_range]) / (center[valid_range] + epsilon)
        smoothed = smooth_func(delta)
        weighted = weights[i] * smoothed

        if i == 0:
            acc = np.zeros_like(prices, dtype=np.float64)
        acc[valid_range] += weighted

    # Оставим значения только в валидной зоне (вокруг которых были данные)
    adc[w_max: n - w_max] = acc[w_max: n - w_max]

    return adc


def plot_prices_and_adc(prices, adc, start=0, end=None):
    if end is None:
        end = len(prices)

    prices_slice = prices[start:end]
    adc_slice = adc[start:end]

    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axs[0].plot(np.arange(start, end), prices_slice, label="Price")
    axs[0].set_title(f"Price Time Series (Index {start}:{end})")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(np.arange(start, end), adc_slice, label="Aggregated Directional Change", color='orange')
    axs[1].axhline(0, color="red", linestyle="--")
    axs[1].set_title("ADC")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    
    symbol = input("Target symbol: ")
    if not symbol:
        symbol = "BTC"
    
    prices = pd.read_csv(f"./data/prices/{symbol}.csv")
    prices = prices["close"]

    adc = aggregated_directional_change(
        prices,
        w_min=w_min,
        w_max=w_max,
        alpha_func=alpha_func,
        smooth_func=smooth_func,
        # smooth_func=lambda x: x,
        epsilon=epsilon,
        use_log=use_log
    )

    # plot_prices_and_adc(prices, adc, 3000, 4000)

    adc.to_csv(f"./data/adc/{symbol}_{w_min}-{w_max}.csv", index=False, header=["adc"])
    
