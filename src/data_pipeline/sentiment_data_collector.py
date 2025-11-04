"""
Sentiment Data Collection Module
Handles collecting and processing financial news for sentiment analysis.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
import os
import time
import logging
from typing import List, Dict, Optional
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentDataCollector:
    """Collects financial news and performs sentiment analysis."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self._setup_logging()
        
        # Financial news RSS feeds
        self.news_feeds = {
            'reuters_forex': 'https://feeds.reuters.com/reuters/UKforex',
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'reuters_economy': 'https://feeds.reuters.com/reuters/economicNews',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/marketpulse/',
            'marketwatch_economy': 'https://feeds.marketwatch.com/marketwatch/economy/',
            'investing_forex': 'https://www.investing.com/rss/news_301.rss',
            'investing_economy': 'https://www.investing.com/rss/news_95.rss',
            'investing_central_banks': 'https://www.investing.com/rss/news_285.rss',
            'forexfactory': 'https://www.forexfactory.com/rss.xml',
            'dailyfx': 'https://www.dailyfx.com/feeds/market-news',
            'fxstreet': 'https://www.fxstreet.com/rss/',
            'bloomberg_forex': 'https://feeds.bloomberg.com/markets/currencies.rss',
            'cnn_economy': 'http://rss.cnn.com/rss/money_latest.rss',
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline'
        }
        
    def _setup_logging(self):
        """Set up logging for the sentiment collector."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def fetch_rss_news(self, feed_url: str, max_articles: int = 50) -> List[Dict]:
        """
        Fetch news articles from RSS feed.
        
        Args:
            feed_url: RSS feed URL
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of article dictionaries
        """
        articles = []
        
        try:
            self.logger.info(f"Fetching news from: {feed_url}")
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:max_articles]:
                article = {
                    'title': entry.get('title', ''),
                    'description': entry.get('description', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'source': feed_url
                }
                
                # Parse publication date
                try:
                    if article['published']:
                        article['published_date'] = pd.to_datetime(entry.published)
                    else:
                        article['published_date'] = datetime.now()
                except:
                    article['published_date'] = datetime.now()
                
                articles.append(article)
                
            self.logger.info(f"Fetched {len(articles)} articles")
            
        except Exception as e:
            self.logger.error(f"Error fetching RSS feed {feed_url}: {e}")
            
        return articles
    
    def collect_all_news(self, max_articles_per_feed: int = 50) -> pd.DataFrame:
        """
        Collect news from all configured RSS feeds.
        
        Args:
            max_articles_per_feed: Maximum articles per feed
            
        Returns:
            DataFrame with all collected articles
        """
        all_articles = []
        
        for feed_name, feed_url in self.news_feeds.items():
            self.logger.info(f"Collecting from {feed_name}")
            articles = self.fetch_rss_news(feed_url, max_articles_per_feed)
            
            # Add feed name to articles
            for article in articles:
                article['feed_name'] = feed_name
                
            all_articles.extend(articles)
            
            # Be respectful to servers
            time.sleep(1)
        
        if all_articles:
            df = pd.DataFrame(all_articles)
            df['published_date'] = pd.to_datetime(df['published_date'])
            df = df.sort_values('published_date', ascending=False)
            df = df.drop_duplicates(subset=['title', 'description'], keep='first')
            
            self.logger.info(f"Collected {len(df)} unique articles")
            return df
        else:
            self.logger.warning("No articles collected")
            return pd.DataFrame()
    
    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        scores = self.vader_analyzer.polarity_scores(text)
        return {
            'vader_positive': scores['pos'],
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu'],
            'vader_compound': scores['compound']
        }
    
    def analyze_sentiment_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        blob = TextBlob(text)
        return {
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity
        }
    
    def process_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment analysis to news DataFrame.
        
        Args:
            df: DataFrame with news articles
            
        Returns:
            DataFrame with sentiment scores added
        """
        if df.empty:
            return df
            
        self.logger.info("Processing sentiment analysis...")
        
        # Combine title and description for analysis
        df['text_content'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
        
        # Apply sentiment analysis
        vader_scores = df['text_content'].apply(self.analyze_sentiment_vader)
        textblob_scores = df['text_content'].apply(self.analyze_sentiment_textblob)
        
        # Convert to DataFrame and join
        vader_df = pd.DataFrame(vader_scores.tolist())
        textblob_df = pd.DataFrame(textblob_scores.tolist())
        
        # Combine all sentiment scores
        result_df = pd.concat([df.reset_index(drop=True), 
                              vader_df.reset_index(drop=True), 
                              textblob_df.reset_index(drop=True)], axis=1)
        
        self.logger.info("Sentiment analysis completed")
        return result_df
    
    def save_sentiment_data(self, data: pd.DataFrame, filename: str = None) -> str:
        """
        Save sentiment data to CSV file.
        
        Args:
            data: DataFrame containing sentiment data
            filename: Custom filename (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        if data.empty:
            self.logger.warning("No sentiment data to save")
            return ""
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_data_{timestamp}.csv"
            
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            data.to_csv(filepath, index=False)
            self.logger.info(f"Sentiment data saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving sentiment data: {e}")
            return ""

if __name__ == "__main__":
    # Example usage
    collector = SentimentDataCollector()
    
    # Collect news and analyze sentiment
    news_data = collector.collect_all_news(max_articles_per_feed=20)
    
    if not news_data.empty:
        # Process sentiment
        sentiment_data = collector.process_sentiment(news_data)
        
        # Save the data
        saved_file = collector.save_sentiment_data(sentiment_data, "forex_sentiment.csv")
        print(f"Sentiment data collected and saved to: {saved_file}")
        print(f"Data shape: {sentiment_data.shape}")
        print(f"Average sentiment (VADER compound): {sentiment_data['vader_compound'].mean():.3f}")
    else:
        print("Failed to collect sentiment data")