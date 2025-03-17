import os
import pandas as pd


class StockDataLoader:
    def __init__(self, data_folder="../src/data/yfinance_data"):
        """
        Initializes the StockDataLoader with the folder path containing the stock data CSV files.

        Args:
            data_folder (str): The path to the folder containing stock data CSV files.
        """
        self.data_folder = data_folder
        self.stock_data_dict = {}

    def load_all_stock_data(self):
        """
        Loads all stock price data from CSV files in the specified data folder.

        Returns:
            dict: A dictionary where the key is the stock ticker (e.g., 'AAPL') and the value is the DataFrame of stock data.
        """
        for file in os.listdir(self.data_folder):
            if file.endswith(".csv"):
                ticker = file.split("_")[0]  # Extract stock ticker (e.g., 'AAPL' from 'AAPL_xxx_data.csv')
                stock_data = pd.read_csv(os.path.join(self.data_folder, file), parse_dates=["Date"], index_col="Date")
                self.stock_data_dict[ticker] = stock_data  # Store the stock data in the dictionary
        return self.stock_data_dict