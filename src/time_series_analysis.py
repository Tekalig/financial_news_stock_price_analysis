import pandas as pd
import matplotlib.pyplot as plt


def plot_stock_data(stock_data, ticker):
    """
    Plot the closing price of a stock over time.

    Args:
        stock_data (pd.DataFrame): The stock data containing 'Close' prices.
        ticker (str): The stock ticker (e.g., 'AAPL').
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    stock_data["Close"].plot(ax=ax, label=f"{ticker} Close Price", color="blue")
    ax.set_title(f"{ticker} Stock Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.grid()
    ax.legend()
    return fig


def plot_time_series(data, title, xlabel, ylabel):
    """
    Plot a time series of data (e.g., publication frequency).

    Args:
        data (pd.Series): The data to plot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(kind="line", marker="o", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    return fig


class TimeSeriesAnalyzer:
    def __init__(self, news_data):
        """
        Initializes the TimeSeriesAnalyzer with news data.

        Args:
            news_data (pd.DataFrame): The news data containing the 'date' column.
        """
        self.news_data = news_data

    def analyze_publication_frequency(self):
        """
        Analyze publication frequency over time (by date).

        Returns:
            pd.Series: A series showing the number of articles published per day.
        """
        self.news_data["date"] = pd.to_datetime(self.news_data["date"], errors="coerce")
        publication_frequency = self.news_data.groupby(
            self.news_data["date"].dt.date
        ).size()
        return publication_frequency

    def analyze_publishing_times(self):
        """
        Analyze the publishing times of news articles (by hour of the day).

        Returns:
            pd.Series: A series showing the number of articles published per hour.
        """
        self.news_data["hour"] = pd.to_datetime(self.news_data["date"]).dt.hour
        publishing_times = self.news_data["hour"].value_counts().sort_index()
        return publishing_times
