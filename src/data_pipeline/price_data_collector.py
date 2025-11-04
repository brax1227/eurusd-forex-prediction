"""
EUR/USD Price Data Collection Module
Handles fetching and storing historical EUR/USD price data from multiple sources.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, Any
import logging

class PriceDataCollector:
    """Collects EUR/USD price data from various financial data providers."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.symbol = "EURUSD=X"
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for the price data collector."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def fetch_yahoo_data(self, 
                        start_date: str = "2020-01-01", 
                        end_date: Optional[str] = None,
                        interval: str = "1d") -> pd.DataFrame:
        """
        Fetch EUR/USD data from Yahoo Finance.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)
            interval: Data interval (1d, 1h, 5m, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        try:
            self.logger.info(f"Fetching EUR/USD data from {start_date} to {end_date}")
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                self.logger.warning("No data returned from Yahoo Finance")
                return pd.DataFrame()
                
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            data.reset_index(inplace=True)
            
            # Handle the date column - could be 'Date' or already be the index
            if 'Date' in data.columns:
                data.rename(columns={'Date': 'date'}, inplace=True)
            elif data.index.name == 'Date' or hasattr(data.index, 'date'):
                data['date'] = data.index
                data.reset_index(drop=True, inplace=True)
            
            data['date'] = pd.to_datetime(data['date'])
            
            self.logger.info(f"Successfully fetched {len(data)} records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo Finance data: {e}")
            return pd.DataFrame()
    
    def save_price_data(self, data: pd.DataFrame, filename: str = None) -> str:
        """
        Save price data to CSV file.
        
        Args:
            data: DataFrame containing price data
            filename: Custom filename (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        if data.empty:
            self.logger.warning("No data to save")
            return ""
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eurusd_price_data_{timestamp}.csv"
            
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            data.to_csv(filepath, index=False)
            self.logger.info(f"Price data saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving price data: {e}")
            return ""
    
    def load_price_data(self, filepath: str) -> pd.DataFrame:
        """
        Load price data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with price data
        """
        try:
            data = pd.read_csv(filepath)
            data['date'] = pd.to_datetime(data['date'])
            self.logger.info(f"Loaded {len(data)} price records from {filepath}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading price data: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    collector = PriceDataCollector()
    
    # Fetch recent EUR/USD data
    price_data = collector.fetch_yahoo_data(start_date="2023-01-01")
    
    if not price_data.empty:
        # Save the data
        saved_file = collector.save_price_data(price_data, "eurusd_historical.csv")
        print(f"Price data collected and saved to: {saved_file}")
        print(f"Data shape: {price_data.shape}")
        print(f"Date range: {price_data['date'].min()} to {price_data['date'].max()}")
    else:
        print("Failed to collect price data")