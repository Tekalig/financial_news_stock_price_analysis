import pandas as pd

def align_datasets(stock_data, news_data):
    """
    Align stock and news datasets by date.

    Args:
        stock_data (pd.DataFrame): Stock price data with a DatetimeIndex.
        news_data (pd.DataFrame): News data with a DatetimeIndex.

    Returns:
        pd.DataFrame: Merged dataset with aligned dates.
    """
    # Ensure both datasets have a DatetimeIndex
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        raise ValueError("stock_data must have a DatetimeIndex.")
    if not isinstance(news_data.index, pd.DatetimeIndex):
        raise ValueError("news_data must have a DatetimeIndex.")

    # Make both indexes timezone-naive
    stock_data.index = stock_data.index.tz_localize('utc')
    news_data.index = news_data.index.tz_localize('utc')
    # Merge datasets on the index
    merged_data = pd.merge(stock_data, news_data, left_index=True, right_index=True, how='inner')

    return merged_data


def calculate_daily_returns(stock_data):
    """
    Calculate daily percentage changes in stock prices.

    Args:
        stock_data (pd.DataFrame): Stock price data with a 'Close' column.

    Returns:
        pd.DataFrame: Stock data with an added 'daily_return' column.
    """
    stock_data['daily_return'] = stock_data['Close'].pct_change()
    return stock_data

def calculate_correlation(merged_data):
    """
    Calculate the Pearson correlation coefficient between sentiment scores
    and daily stock returns.

    Args:
        merged_data (pd.DataFrame): Merged dataset with 'sentiment_score' and 'daily_return'.

    Returns:
        float: Pearson correlation coefficient.
    """
    # Aggregate sentiment scores by date
    daily_sentiment = merged_data.groupby('date')['sentiment'].mean()
    merged_data = merged_data.set_index('date')
    merged_data['average_sentiment'] = daily_sentiment

    # Calculate correlation
    correlation = merged_data['average_sentiment'].corr(merged_data['daily_return'])
    return correlation
