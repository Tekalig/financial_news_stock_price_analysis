import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer


class SentimentAnalyzer:
    def __init__(self, news_data):
        """
        Initializes the SentimentAnalyzer with news data.

        Args:
            news_data (pd.DataFrame): The news data containing the 'headline' column.
        """
        self.news_data = news_data

    def perform_sentiment_analysis(self):
        """
        Perform sentiment analysis on the 'headline' column using TextBlob.

        Returns:
            pd.DataFrame: The news data with sentiment scores and categories.
        """
        # Calculate sentiment polarity using TextBlob
        self.news_data['sentiment'] = self.news_data['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
        # Classify sentiment into positive, negative, or neutral
        self.news_data['sentiment_category'] = self.news_data['sentiment'].apply(
            lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral'
        )
        return self.news_data

    def extract_keywords(self, max_features=10):
        """
        Extract the most common keywords from the 'headline' column using CountVectorizer.

        Args:
            max_features (int): The maximum number of keywords to extract.

        Returns:
            list: A list of the most common keywords.
        """
        vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        keywords_matrix = vectorizer.fit_transform(self.news_data['headline'])
        keywords = vectorizer.get_feature_names_out()
        return keywords