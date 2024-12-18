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


def plot_sentiment_vs_returns(merged_data):
    """
    Plot average sentiment scores against daily stock returns.

    Args:
        merged_data (pd.DataFrame): Merged dataset with 'average_sentiment' and 'daily_return'.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_data['average_sentiment'], merged_data['daily_return'], alpha=0.5)
    plt.title('Sentiment vs. Stock Returns')
    plt.xlabel('Average Sentiment Score')
    plt.ylabel('Daily Stock Return')
    plt.grid()
    plt.show()
