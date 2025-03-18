import pandas as pd
import matplotlib.pyplot as plt
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
        self.news_data["sentiment"] = self.news_data["headline"].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        # Classify sentiment into positive, negative, or neutral
        self.news_data["sentiment_category"] = self.news_data["sentiment"].apply(
            lambda x: "positive" if x > 0 else "negative" if x < 0 else "neutral"
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
        vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        keywords_matrix = vectorizer.fit_transform(self.news_data["headline"])
        keywords = vectorizer.get_feature_names_out()
        return keywords

    def pi_chart_sentiment_distribution(self):
        """
        Display a pie chart showing the distribution of sentiment categories.
        """
        sentiment_distribution = self.news_data["sentiment_category"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            sentiment_distribution,
            labels=sentiment_distribution.index,
            autopct="%1.1f%%",
            startangle=140,
        )
        ax.set_title("Sentiment Distribution of News Headlines")
        return fig

    def plot_publisher_activity(self):
        """
        Plot the number of articles published by different publishers over time.

        Args:
            news_data (pd.DataFrame): News data with 'publisher' and 'date' columns.
        """
        self.news_data["date"] = pd.to_datetime(self.news_data["date"], errors="coerce")

        # Count the total number of articles per publisher
        publisher_counts = self.news_data["publisher"].value_counts().index

        # Filter for top publishers
        filtered_data = self.news_data[
            self.news_data["publisher"].isin(publisher_counts)
        ]

        # Group by date and publisher, and aggregate counts
        grouped = (
            filtered_data.groupby([pd.Grouper(key="date", freq="M"), "publisher"])
            .size()
            .reset_index(name="article_count")
        )

        # Pivot the data for plotting
        pivot_table = grouped.pivot(
            index="date", columns="publisher", values="article_count"
        ).fillna(0)

        # Plot a grouped bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        pivot_table.plot(kind="bar", stacked=False, width=0.8, ax=ax)

        ax.set_title("Publisher Activity Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Articles")
        ax.legend(title="Publisher", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        return fig

    def calculate_publisher_sentiment(self):
        """
        Calculate average sentiment scores for each publisher.

        Args:
            news_data (pd.DataFrame): News data with 'publisher' and 'sentiment_score' columns.

        Returns:
            pd.DataFrame: Publisher-wise sentiment scores with counts and averages.
        """
        # Group by publisher and calculate sentiment statistics
        sentiment_summary = (
            self.news_data.groupby("publisher")["sentiment"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "average_sentiment", "count": "article_count"})
        )

        return sentiment_summary

    def plot_publisher_sentiment(self, sentiment_summary, top_n=10):
        """
        Plot average sentiment scores for top publishers.

        Args:
            sentiment_summary (pd.DataFrame): DataFrame with 'publisher', 'average_sentiment', and 'article_count'.
            top_n (int): Number of top publishers to display based on article count.
        """
        # Sort by article count and select top publishers
        top_publishers = sentiment_summary.nlargest(top_n, "article_count")

        # Create a bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(
            top_publishers["publisher"],
            top_publishers["average_sentiment"],
            color="skyblue",
        )
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"Publisher Sentiment Analysis (Top {top_n} Publishers)")
        ax.set_xlabel("Publisher")
        ax.set_ylabel("Average Sentiment Score")
        ax.set_xticks(range(len(top_publishers["publisher"])))
        ax.set_xticklabels(top_publishers["publisher"], rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        return fig
