import pandas as pd


def load_stock_data(filepath):
    """
    Load stock price data from a CSV file.

    Args:
        filepath (str): Path to the stock price CSV file.

    Returns:
        pd.DataFrame: Cleaned stock price data with datetime index.
    """
    try:
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        return data
    except Exception as e:
        print(f"Error loading stock data: {e}")
        return None
