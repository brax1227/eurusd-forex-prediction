"""
Enhanced News Collection Module
Implements multiple strategies to collect more comprehensive financial news data.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
import os
import time
import logging
import json
from typing import List, Dict, Optional
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import concurrent.futures
from urllib.parse import urljoin, urlparse
import random

class EnhancedNewsCollector:
    """Enhanced financial news collector with multiple data sources and strategies."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self._setup_logging()
        
        # Extended list of financial news sources
        self.news_sources = {
            # Major Financial News
            'reuters_forex': 'https://feeds.reuters.com/reuters/UKforex',
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'reuters_economy': 'https://feeds.reuters.com/reuters/economicNews',
            'reuters_worldnews': 'https://feeds.reuters.com/reuters/worldNews',
            
            # Market Specific
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/marketpulse/',
            'marketwatch_economy': 'https://feeds.marketwatch.com/marketwatch/economy/',
            'marketwatch_europe': 'https://feeds.marketwatch.com/marketwatch/europe/',
            
            # Forex Specialized
            'investing_forex': 'https://www.investing.com/rss/news_301.rss',
            'investing_economy': 'https://www.investing.com/rss/news_95.rss',
            'investing_central_banks': 'https://www.investing.com/rss/news_285.rss',
            'forexfactory': 'https://www.forexfactory.com/rss.xml',
            'dailyfx': 'https://www.dailyfx.com/feeds/market-news',
            'fxstreet': 'https://www.fxstreet.com/rss/',
            
            # Major Financial Media
            'bloomberg_markets': 'https://feeds.bloomberg.com/markets/news.rss',
            'bloomberg_economics': 'https://feeds.bloomberg.com/economics/news.rss',
            'cnn_economy': 'http://rss.cnn.com/rss/money_latest.rss',
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'cnbc_forex': 'https://www.cnbc.com/id/15839135/device/rss/rss.html',
            'cnbc_world': 'https://www.cnbc.com/id/100727362/device/rss/rss.html',
            
            # Economic Data
            'fed_news': 'https://www.federalreserve.gov/feeds/press_all.xml',
            'ecb_news': 'https://www.ecb.europa.eu/rss/press.html',
            
            # Additional Sources
            'financial_times': 'https://www.ft.com/rss/home',
            'wsj_economy': 'https://feeds.a.dj.com/rss/RSSEconomics.xml',
            'business_insider': 'https://feeds.businessinsider.com/businessinsider',
        }
        
        # Keywords for filtering relevant forex/economic news
        self.forex_keywords = [
            'euro', 'dollar', 'eur', 'usd', 'eurusd', 'eur/usd', 'currency',
            'exchange rate', 'forex', 'federal reserve', 'fed', 'ecb', 
            'european central bank', 'interest rate', 'monetary policy',
            'inflation', 'gdp', 'unemployment', 'trade', 'economic',
            'central bank', 'draghi', 'lagarde', 'powell', 'yellen'
        ]
        
    def _setup_logging(self):
        """Set up logging for the enhanced collector."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def fetch_rss_with_retry(self, feed_url: str, max_articles: int = 100, retries: int = 3) -> List[Dict]:
        """Fetch RSS feed with retry logic and error handling."""
        articles = []
        
        for attempt in range(retries):
            try:
                self.logger.info(f"Fetching from {feed_url} (attempt {attempt + 1}/{retries})")
                
                # Add user agent to avoid blocking
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                # Parse RSS feed
                feed = feedparser.parse(feed_url, request_headers=headers)
                
                if hasattr(feed, 'entries') and feed.entries:
                    for entry in feed.entries[:max_articles]:
                        article = {
                            'title': entry.get('title', ''),
                            'description': entry.get('description', '') or entry.get('summary', ''),
                            'link': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'source': feed_url,
                            'feed_name': urlparse(feed_url).netloc
                        }
                        
                        # Parse publication date
                        try:
                            if article['published']:
                                article['published_date'] = pd.to_datetime(entry.published)
                            else:
                                article['published_date'] = datetime.now()
                        except:
                            article['published_date'] = datetime.now()
                        
                        # Filter for forex-relevant content
                        if self._is_forex_relevant(article):
                            articles.append(article)
                    
                    self.logger.info(f"Fetched {len(articles)} relevant articles from {feed_url}")
                    break
                else:
                    self.logger.warning(f"No entries found in RSS feed: {feed_url}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching RSS feed {feed_url} (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        return articles
    
    def _is_forex_relevant(self, article: Dict) -> bool:
        """Check if article is relevant to forex/EUR-USD trading."""
        text_content = (article.get('title', '') + ' ' + article.get('description', '')).lower()
        
        # Check for forex keywords
        relevance_score = sum(1 for keyword in self.forex_keywords if keyword in text_content)
        
        # Article is relevant if it contains at least 1 forex keyword
        return relevance_score > 0
    
    def collect_comprehensive_news(self, max_articles_per_feed: int = 200) -> pd.DataFrame:
        """Collect news from all sources using parallel processing."""
        all_articles = []
        
        # Use ThreadPoolExecutor for parallel collection
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all RSS feed tasks
            future_to_feed = {
                executor.submit(self.fetch_rss_with_retry, feed_url, max_articles_per_feed): feed_name
                for feed_name, feed_url in self.news_sources.items()
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_feed):
                feed_name = future_to_feed[future]
                try:
                    articles = future.result()
                    # Add feed name to articles
                    for article in articles:
                        article['feed_name'] = feed_name
                    all_articles.extend(articles)
                    
                    # Add small delay to be respectful
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Error collecting from {feed_name}: {e}")
        
        if all_articles:
            df = pd.DataFrame(all_articles)
            df['published_date'] = pd.to_datetime(df['published_date'], utc=True, errors='coerce')
            df = df.sort_values('published_date', ascending=False)
            
            # Remove duplicates based on title and description
            initial_count = len(df)
            df = df.drop_duplicates(subset=['title', 'description'], keep='first')
            
            self.logger.info(f"Collected {len(df)} unique articles ({initial_count - len(df)} duplicates removed)")
            return df
        else:
            self.logger.warning("No articles collected")
            return pd.DataFrame()
    
    def generate_synthetic_historical_sentiment(self, start_date: str = "2019-01-01") -> pd.DataFrame:
        """
        Generate synthetic historical sentiment data to fill gaps.
        This simulates market sentiment patterns for historical periods.
        """
        self.logger.info("Generating synthetic historical sentiment data...")
        
        # Create date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime('today')
        date_range = pd.date_range(start, end, freq='D')
        
        synthetic_data = []
        
        for date in date_range:
            # Create realistic but synthetic sentiment patterns
            # Base sentiment with some market cycle patterns
            base_sentiment = 0.0
            
            # Add weekly patterns (Monday blues, Friday optimism)
            day_of_week = date.weekday()
            if day_of_week == 0:  # Monday
                base_sentiment -= 0.1
            elif day_of_week == 4:  # Friday
                base_sentiment += 0.05
            
            # Add monthly patterns (beginning vs end of month)
            if date.day <= 5:
                base_sentiment += 0.02
            elif date.day >= 25:
                base_sentiment -= 0.02
            
            # Add some random variation
            noise = random.gauss(0, 0.15)
            compound_score = max(-1, min(1, base_sentiment + noise))
            
            # Derive other sentiment scores from compound
            if compound_score >= 0.05:
                positive = 0.3 + random.random() * 0.4
                negative = 0.0 + random.random() * 0.2
            elif compound_score <= -0.05:
                positive = 0.0 + random.random() * 0.2
                negative = 0.3 + random.random() * 0.4
            else:
                positive = 0.1 + random.random() * 0.3
                negative = 0.1 + random.random() * 0.3
            
            neutral = max(0, 1 - positive - negative)
            
            synthetic_data.append({
                'published_date': date,
                'title': f'Synthetic market sentiment for {date.strftime("%Y-%m-%d")}',
                'description': 'Generated historical sentiment data',
                'source': 'synthetic',
                'feed_name': 'synthetic_sentiment',
                'vader_compound': compound_score,
                'vader_positive': positive,
                'vader_negative': negative,
                'vader_neutral': neutral,
                'textblob_polarity': compound_score * 0.8,  # Similar but slightly different
                'textblob_subjectivity': 0.3 + random.random() * 0.4
            })
        
        df = pd.DataFrame(synthetic_data)
        self.logger.info(f"Generated {len(df)} synthetic sentiment records")
        return df
    
    def analyze_sentiment_enhanced(self, text: str) -> Dict[str, float]:
        """Enhanced sentiment analysis with additional features."""
        vader_scores = self.vader_analyzer.polarity_scores(text)
        blob = TextBlob(text)
        
        # Count exclamation marks and question marks as indicators
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Financial terms sentiment modifiers
        bullish_terms = ['bullish', 'optimistic', 'positive', 'growth', 'rise', 'gain', 'strong']
        bearish_terms = ['bearish', 'pessimistic', 'negative', 'decline', 'fall', 'loss', 'weak']
        
        bullish_count = sum(1 for term in bullish_terms if term in text.lower())
        bearish_count = sum(1 for term in bearish_terms if term in text.lower())
        
        return {
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'vader_compound': vader_scores['compound'],
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity,
            'exclamation_intensity': min(exclamation_count / 10.0, 1.0),
            'question_uncertainty': min(question_count / 10.0, 1.0),
            'bullish_terms': bullish_count,
            'bearish_terms': bearish_count,
            'financial_sentiment_bias': (bullish_count - bearish_count) / max(1, bullish_count + bearish_count)
        }
    
    def process_sentiment_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced sentiment processing with additional features."""
        if df.empty:
            return df
            
        self.logger.info("Processing enhanced sentiment analysis...")
        
        # Combine title and description
        df['text_content'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
        
        # Apply enhanced sentiment analysis
        sentiment_results = df['text_content'].apply(self.analyze_sentiment_enhanced)
        
        # Convert to DataFrame and join
        sentiment_df = pd.DataFrame(sentiment_results.tolist())
        result_df = pd.concat([df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)
        
        self.logger.info("Enhanced sentiment analysis completed")
        return result_df
    
    def save_enhanced_data(self, data: pd.DataFrame, filename: str = None) -> str:
        """Save enhanced sentiment data."""
        if data.empty:
            self.logger.warning("No data to save")
            return ""
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_sentiment_data_{timestamp}.csv"
            
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            data.to_csv(filepath, index=False)
            self.logger.info(f"Enhanced sentiment data saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving enhanced data: {e}")
            return ""

if __name__ == "__main__":
    # Example usage
    collector = EnhancedNewsCollector()
    
    # Collect comprehensive news
    news_data = collector.collect_comprehensive_news(max_articles_per_feed=100)
    
    if not news_data.empty:
        # Process with enhanced sentiment analysis
        enhanced_data = collector.process_sentiment_enhanced(news_data)
        
        # Save the enhanced data
        saved_file = collector.save_enhanced_data(enhanced_data, "comprehensive_forex_sentiment.csv")
        print(f"Enhanced sentiment data saved: {saved_file}")
        print(f"Articles collected: {len(enhanced_data)}")
        print(f"Date range: {enhanced_data['published_date'].min()} to {enhanced_data['published_date'].max()}")
        print(f"Average sentiment: {enhanced_data['vader_compound'].mean():.3f}")
    
    # Generate synthetic historical data to fill gaps
    synthetic_data = collector.generate_synthetic_historical_sentiment("2019-01-01")
    synthetic_file = collector.save_enhanced_data(synthetic_data, "synthetic_sentiment_data.csv")
    print(f"Synthetic historical data saved: {synthetic_file}")