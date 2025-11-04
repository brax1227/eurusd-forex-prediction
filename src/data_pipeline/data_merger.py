"""
Data Merger Module
Combines price data and sentiment data into a unified dataset for ML training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any
import os

class DataMerger:
    """Merges price and sentiment data for ML model training."""
    
    def __init__(self, processed_data_dir: str = "data/processed"):
        self.processed_data_dir = processed_data_dir
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for the data merger."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_price_data(self, filepath: str) -> pd.DataFrame:
        """Load and prepare price data."""
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
            df = df.sort_values('date')
            self.logger.info(f"Loaded {len(df)} price records")
            return df
        except Exception as e:
            self.logger.error(f"Error loading price data: {e}")
            return pd.DataFrame()
    
    def load_sentiment_data(self, filepath: str) -> pd.DataFrame:
        """Load and prepare sentiment data."""
        try:
            df = pd.read_csv(filepath)
            df['published_date'] = pd.to_datetime(df['published_date'], utc=True).dt.tz_localize(None)
            df = df.sort_values('published_date')
            self.logger.info(f"Loaded {len(df)} sentiment records")
            return df
        except Exception as e:
            self.logger.error(f"Error loading sentiment data: {e}")
            return pd.DataFrame()
    
    def aggregate_daily_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment scores by day.
        
        Args:
            sentiment_df: DataFrame with sentiment data
            
        Returns:
            DataFrame with daily aggregated sentiment
        """
        if sentiment_df.empty:
            return pd.DataFrame()
            
        # Extract date from published_date
        sentiment_df['date'] = sentiment_df['published_date'].dt.date
        
        # Define aggregation functions
        agg_functions = {
            'vader_compound': ['mean', 'std', 'min', 'max', 'count'],
            'vader_positive': ['mean', 'std'],
            'vader_negative': ['mean', 'std'],
            'vader_neutral': ['mean'],
            'textblob_polarity': ['mean', 'std', 'min', 'max'],
            'textblob_subjectivity': ['mean', 'std']
        }
        
        # Aggregate by date
        daily_sentiment = sentiment_df.groupby('date').agg(agg_functions)
        
        # Flatten column names
        daily_sentiment.columns = [f"{col[0]}_{col[1]}" for col in daily_sentiment.columns]
        
        # Reset index to make date a column
        daily_sentiment = daily_sentiment.reset_index()
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        # Rename count column for clarity
        if 'vader_compound_count' in daily_sentiment.columns:
            daily_sentiment.rename(columns={'vader_compound_count': 'news_count'}, inplace=True)
        
        self.logger.info(f"Aggregated sentiment data into {len(daily_sentiment)} daily records")
        return daily_sentiment
    
    def calculate_price_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators and price-based features.
        
        Args:
            price_df: DataFrame with OHLCV price data
            
        Returns:
            DataFrame with additional price features
        """
        if price_df.empty:
            return pd.DataFrame()
            
        df = price_df.copy()
        
        # Basic price features
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['open_close_range'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
        
        # Volatility measures
        df['volatility_5d'] = df['price_change'].rolling(window=5).std()
        df['volatility_20d'] = df['price_change'].rolling(window=20).std()
        
        # RSI (simplified)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Target variable: next day price movement direction
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        self.logger.info("Calculated price-based features")
        return df
    
    def merge_price_sentiment(self, 
                             price_df: pd.DataFrame, 
                             sentiment_df: pd.DataFrame,
                             method: str = 'left') -> pd.DataFrame:
        """
        Merge price and sentiment data on date.
        
        Args:
            price_df: DataFrame with price data and features
            sentiment_df: DataFrame with aggregated sentiment data
            method: Merge method ('left', 'inner', 'outer')
            
        Returns:
            Merged DataFrame
        """
        if price_df.empty:
            self.logger.warning("No price data to merge")
            return pd.DataFrame()
            
        if sentiment_df.empty:
            self.logger.warning("No sentiment data to merge, returning price data only")
            return price_df
            
        # Ensure both DataFrames have date columns
        price_df['merge_date'] = pd.to_datetime(price_df['date']).dt.date
        sentiment_df['merge_date'] = pd.to_datetime(sentiment_df['date']).dt.date
        
        # Merge on date
        merged_df = pd.merge(
            price_df, 
            sentiment_df.drop('date', axis=1), 
            on='merge_date', 
            how=method
        )
        
        # Clean up
        merged_df.drop('merge_date', axis=1, inplace=True)
        
        # Fill missing sentiment values with neutral values
        sentiment_columns = [col for col in merged_df.columns if any(x in col for x in ['vader', 'textblob', 'news_count'])]
        for col in sentiment_columns:
            if 'count' in col:
                merged_df[col] = merged_df[col].fillna(0)
            elif 'neutral' in col:
                merged_df[col] = merged_df[col].fillna(1.0)
            else:
                merged_df[col] = merged_df[col].fillna(0.0)
        
        self.logger.info(f"Merged data: {len(merged_df)} records with {len(merged_df.columns)} features")
        return merged_df
    
    def save_merged_data(self, data: pd.DataFrame, filename: str = None) -> str:
        """
        Save merged dataset to CSV file.
        
        Args:
            data: Merged DataFrame
            filename: Custom filename (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        if data.empty:
            self.logger.warning("No data to save")
            return ""
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"merged_dataset_{timestamp}.csv"
            
        os.makedirs(self.processed_data_dir, exist_ok=True)
        filepath = os.path.join(self.processed_data_dir, filename)
        
        try:
            data.to_csv(filepath, index=False)
            self.logger.info(f"Merged dataset saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving merged data: {e}")
            return ""
    
    def create_training_dataset(self, 
                               price_filepath: str, 
                               sentiment_filepath: str,
                               output_filename: str = None) -> str:
        """
        Complete pipeline to create training dataset.
        
        Args:
            price_filepath: Path to price data CSV
            sentiment_filepath: Path to sentiment data CSV
            output_filename: Custom output filename
            
        Returns:
            Path to saved training dataset
        """
        self.logger.info("Starting training dataset creation pipeline...")
        
        # Load data
        price_df = self.load_price_data(price_filepath)
        sentiment_df = self.load_sentiment_data(sentiment_filepath)
        
        if price_df.empty:
            self.logger.error("No price data available")
            return ""
        
        # Calculate price features
        price_with_features = self.calculate_price_features(price_df)
        
        # Aggregate sentiment data by day
        daily_sentiment = self.aggregate_daily_sentiment(sentiment_df)
        
        # Merge datasets
        merged_dataset = self.merge_price_sentiment(price_with_features, daily_sentiment)
        
        if merged_dataset.empty:
            self.logger.error("Failed to create merged dataset")
            return ""
        
        # Save the final dataset
        output_path = self.save_merged_data(merged_dataset, output_filename)
        
        if output_path:
            self.logger.info(f"Training dataset created successfully: {output_path}")
            self.logger.info(f"Dataset shape: {merged_dataset.shape}")
            self.logger.info(f"Features: {list(merged_dataset.columns)}")
            
        return output_path

if __name__ == "__main__":
    # Example usage
    merger = DataMerger()
    
    # Assuming you have price and sentiment data files
    price_file = "data/raw/eurusd_historical.csv"
    sentiment_file = "data/raw/forex_sentiment.csv"
    
    # Create training dataset
    training_dataset_path = merger.create_training_dataset(
        price_file, 
        sentiment_file, 
        "eurusd_training_dataset.csv"
    )
    
    if training_dataset_path:
        print(f"Training dataset created: {training_dataset_path}")
    else:
        print("Failed to create training dataset")