import talib

def calculate_moving_averages(data, period=30):
    """
    Calculate Simple Moving Average (SMA) for the given period.

    Args:
        data (pd.DataFrame): Stock price data with a 'Close' column.
        period (int): Period for the moving average.

    Returns:
        pd.Series: SMA values.
    """
    return talib.SMA(data['Close'], timeperiod=period)

def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI) for the given period.

    Args:
        data (pd.DataFrame): Stock price data with a 'Close' column.
        period (int): Period for RSI.

    Returns:
        pd.Series: RSI values.
    """
    return talib.RSI(data['Close'], timeperiod=period)

def calculate_macd(data):
    """
    Calculate Moving Average Convergence Divergence (MACD).

    Args:
        data (pd.DataFrame): Stock price data with a 'Close' column.

    Returns:
        tuple: MACD, Signal Line, Histogram.
    """
    macd, macd_signal, macd_hist = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return macd, macd_signal, macd_hist
