import matplotlib.pyplot as plt

def plot_moving_averages(data, sma):
    """
    Plot stock prices with SMA overlay.

    Args:
        data (pd.DataFrame): Stock price data.
        sma (pd.Series): Simple Moving Average.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.plot(data.index, sma, label='SMA (30 days)', color='orange', linestyle='--')
    plt.title('Stock Prices with SMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def plot_macd(data, macd, macd_signal, macd_hist):
    """
    Plot MACD indicators.

    Args:
        data (pd.DataFrame): Stock price data.
        macd (pd.Series): MACD line.
        macd_signal (pd.Series): Signal line.
        macd_hist (pd.Series): Histogram.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, macd, label='MACD', color='blue')
    plt.plot(data.index, macd_signal, label='Signal Line', color='red', linestyle='--')
    plt.bar(data.index, macd_hist, label='Histogram', color='gray', alpha=0.5)
    plt.title('MACD Indicators')
    plt.xlabel('Date')
    plt.legend()
    plt.grid()
    plt.show()
