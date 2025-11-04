"""
Data Cleaning and Formatting Module
Prepares the comprehensive dataset for ML modeling with proper formatting and documentation.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

class DataCleaner:
    """Clean and format the comprehensive dataset for ML modeling."""
    
    def __init__(self):
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for the data cleaner."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_and_clean_dataset(self, input_path: str) -> pd.DataFrame:
        """Load and clean the comprehensive dataset."""
        self.logger.info(f"Loading dataset from {input_path}")
        
        # Load the dataset
        df = pd.read_csv(input_path)
        initial_shape = df.shape
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date to ensure chronological order
        df = df.sort_values('date').reset_index(drop=True)
        
        self.logger.info(f"Loaded dataset with shape: {initial_shape}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'smart') -> pd.DataFrame:
        """Handle missing values intelligently."""
        self.logger.info("Handling missing values...")
        
        df_clean = df.copy()
        initial_missing = df_clean.isnull().sum().sum()
        
        if method == 'smart':
            # Technical indicators - forward fill then drop initial NaN rows
            tech_cols = [col for col in df_clean.columns if any(x in col for x in ['ma_', 'rsi', 'volatility'])]
            for col in tech_cols:
                df_clean[col] = df_clean[col].fillna(method='ffill')
            
            # Sentiment features - fill with neutral/zero values
            sentiment_cols = [col for col in df_clean.columns if any(x in col for x in ['vader', 'textblob'])]
            for col in sentiment_cols:
                if 'news_count' in col:
                    df_clean[col] = df_clean[col].fillna(0)
                elif 'neutral' in col:
                    df_clean[col] = df_clean[col].fillna(1.0)
                elif 'std' in col:
                    df_clean[col] = df_clean[col].fillna(0.0)
                else:
                    df_clean[col] = df_clean[col].fillna(0.0)
            
            # Price changes - can be calculated if missing
            if df_clean['price_change'].isnull().any():
                df_clean['price_change'] = df_clean['close'].pct_change()
                df_clean['price_change_abs'] = df_clean['price_change'].abs()
        
        elif method == 'drop':
            # Drop rows with any missing values
            df_clean = df_clean.dropna()
        
        elif method == 'forward_fill':
            # Forward fill all missing values
            df_clean = df_clean.fillna(method='ffill')
        
        final_missing = df_clean.isnull().sum().sum()
        self.logger.info(f"Missing values: {initial_missing} → {final_missing}")
        
        return df_clean
    
    def add_feature_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata and ensure proper data types."""
        self.logger.info("Adding feature metadata and formatting...")
        
        df_formatted = df.copy()
        
        # Ensure proper data types
        df_formatted['date'] = pd.to_datetime(df_formatted['date'])
        df_formatted['target'] = df_formatted['target'].astype(int)
        
        # Round numeric columns to reasonable precision
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df_formatted[col] = df_formatted[col].round(6)
        
        # Round ratios and percentages
        ratio_cols = [col for col in df_formatted.columns if 'ratio' in col or 'change' in col]
        for col in ratio_cols:
            df_formatted[col] = df_formatted[col].round(6)
        
        # Round sentiment scores
        sentiment_cols = [col for col in df_formatted.columns if any(x in col for x in ['vader', 'textblob'])]
        for col in sentiment_cols:
            if 'count' not in col:
                df_formatted[col] = df_formatted[col].round(4)
        
        return df_formatted
    
    def create_feature_dictionary(self, df: pd.DataFrame) -> dict:
        """Create a comprehensive feature dictionary."""
        feature_dict = {
            'dataset_info': {
                'name': 'EUR/USD Forex Prediction Dataset',
                'shape': df.shape,
                'date_range': f"{df['date'].min()} to {df['date'].max()}",
                'created': datetime.now().isoformat(),
                'target_balance': df['target'].value_counts().to_dict()
            },
            'features': {}
        }
        
        # Categorize features
        categories = {
            'date': ['date'],
            'price': ['open', 'high', 'low', 'close', 'volume'],
            'technical': [col for col in df.columns if any(x in col for x in ['ma_', 'rsi', 'volatility', 'price_change', 'range'])],
            'sentiment': [col for col in df.columns if any(x in col for x in ['vader', 'textblob', 'news'])],
            'target': ['target'],
            'other': ['dividends', 'stock_splits']
        }
        
        # Add feature descriptions
        descriptions = {
            'date': 'Trading date in YYYY-MM-DD format',
            'open': 'Opening EUR/USD exchange rate',
            'high': 'Highest EUR/USD exchange rate for the day',
            'low': 'Lowest EUR/USD exchange rate for the day', 
            'close': 'Closing EUR/USD exchange rate',
            'volume': 'Trading volume (always 0 for forex)',
            'price_change': 'Daily price change percentage',
            'price_change_abs': 'Absolute daily price change',
            'high_low_range': '(High - Low) / Close ratio',
            'open_close_range': '(Close - Open) / Open ratio',
            'ma_5': '5-day moving average',
            'ma_10': '10-day moving average',
            'ma_20': '20-day moving average',
            'ma_50': '50-day moving average',
            'close_ma_5_ratio': 'Close price / 5-day MA ratio',
            'close_ma_10_ratio': 'Close price / 10-day MA ratio',
            'close_ma_20_ratio': 'Close price / 20-day MA ratio',
            'close_ma_50_ratio': 'Close price / 50-day MA ratio',
            'volatility_5d': '5-day rolling volatility',
            'volatility_20d': '20-day rolling volatility',
            'rsi': 'Relative Strength Index (14-day)',
            'news_count': 'Number of news articles/events for the day',
            'vader_compound_mean': 'Average VADER compound sentiment (-1 to +1)',
            'vader_compound_std': 'Standard deviation of VADER compound sentiment',
            'vader_compound_min': 'Minimum VADER compound sentiment',
            'vader_compound_max': 'Maximum VADER compound sentiment',
            'vader_positive_mean': 'Average VADER positive sentiment score',
            'vader_positive_std': 'Standard deviation of VADER positive sentiment',
            'vader_negative_mean': 'Average VADER negative sentiment score',
            'vader_negative_std': 'Standard deviation of VADER negative sentiment',
            'vader_neutral_mean': 'Average VADER neutral sentiment score',
            'textblob_polarity_mean': 'Average TextBlob polarity (-1 to +1)',
            'textblob_polarity_std': 'Standard deviation of TextBlob polarity',
            'textblob_polarity_min': 'Minimum TextBlob polarity',
            'textblob_polarity_max': 'Maximum TextBlob polarity',
            'textblob_subjectivity_mean': 'Average TextBlob subjectivity (0 to 1)',
            'textblob_subjectivity_std': 'Standard deviation of TextBlob subjectivity',
            'target': 'Binary target: 0=price down, 1=price up next day',
            'dividends': 'Dividend payments (always 0 for forex)',
            'stock_splits': 'Stock splits (always 0 for forex)'
        }
        
        # Build feature dictionary
        for category, columns in categories.items():
            feature_dict['features'][category] = {}
            for col in columns:
                if col in df.columns:
                    feature_dict['features'][category][col] = {
                        'description': descriptions.get(col, 'No description available'),
                        'dtype': str(df[col].dtype),
                        'missing_values': int(df[col].isnull().sum()),
                        'unique_values': int(df[col].nunique()) if df[col].dtype in ['object', 'int64'] else None
                    }
                    
                    # Add statistics for numeric columns
                    if df[col].dtype in ['float64', 'int64'] and col != 'date':
                        stats = df[col].describe()
                        feature_dict['features'][category][col]['statistics'] = {
                            'mean': round(float(stats['mean']), 4),
                            'std': round(float(stats['std']), 4),
                            'min': round(float(stats['min']), 4),
                            'max': round(float(stats['max']), 4)
                        }
        
        return feature_dict
    
    def save_cleaned_dataset(self, df: pd.DataFrame, output_path: str, feature_dict: dict = None) -> str:
        """Save the cleaned dataset with proper formatting."""
        self.logger.info(f"Saving cleaned dataset to {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save main dataset
        df.to_csv(output_path, index=False, float_format='%.6f')
        
        # Save feature dictionary if provided
        if feature_dict:
            dict_path = output_path.replace('.csv', '_features.json')
            import json
            with open(dict_path, 'w') as f:
                json.dump(feature_dict, f, indent=2, default=str)
            self.logger.info(f"Feature dictionary saved to {dict_path}")
        
        # Generate summary
        summary = {
            'file_path': output_path,
            'shape': df.shape,
            'size_mb': round(os.path.getsize(output_path) / 1024**2, 2),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'missing_values': int(df.isnull().sum().sum()),
            'target_distribution': df['target'].value_counts().to_dict(),
            'news_coverage': f"{(df['news_count'] > 0).sum()}/{len(df)} days ({(df['news_count'] > 0).mean()*100:.1f}%)"
        }
        
        self.logger.info(f"Dataset saved successfully: {summary}")
        return output_path
    
    def create_sample_notebook(self, df: pd.DataFrame, output_dir: str):
        """Create a sample Jupyter notebook for data exploration."""
        notebook_content = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# EUR/USD Forex Prediction Dataset - Data Exploration\\n",
    "\\n",
    "This notebook provides a quick start guide for exploring the comprehensive EUR/USD dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from datetime import datetime\\n",
    "\\n",
    "# Set display options\\n",
    "pd.set_option('display.max_columns', None)\\n",
    "plt.style.use('default')\\n",
    "\\n",
    "print('📊 EUR/USD Forex Dataset Explorer')\\n",
    "print('=' * 50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\\n",
    "df = pd.read_csv('../data/processed/ml_ready_dataset.csv')\\n",
    "df['date'] = pd.to_datetime(df['date'])\\n",
    "\\n",
    "print(f'Dataset Shape: {df.shape}')\\n",
    "print(f'Date Range: {df[\\"date\\"].min()} to {df[\\"date\\"].max()}')\\n",
    "print(f'Missing Values: {df.isnull().sum().sum()}')\\n",
    "\\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic statistics\\n",
    "print('🎯 Target Distribution:')\\n",
    "print(df['target'].value_counts())\\n",
    "print(f'\\\\nBalance Ratio: {min(df[\\"target\\"].value_counts()) / max(df[\\"target\\"].value_counts()):.3f}')\\n",
    "\\n",
    "print('\\\\n📰 News Coverage:')\\n",
    "news_days = (df['news_count'] > 0).sum()\\n",
    "print(f'Days with news: {news_days:,}/{len(df):,} ({news_days/len(df)*100:.1f}%)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature categories\\n",
    "price_features = ['open', 'high', 'low', 'close']\\n",
    "technical_features = [col for col in df.columns if any(x in col for x in ['ma_', 'rsi', 'volatility', 'price_change'])]\\n",
    "sentiment_features = [col for col in df.columns if any(x in col for x in ['vader', 'textblob', 'news'])]\\n",
    "\\n",
    "print(f'📈 Technical Features ({len(technical_features)}): {technical_features[:5]}...')\\n",
    "print(f'📰 Sentiment Features ({len(sentiment_features)}): {sentiment_features[:5]}...')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
        
        notebook_path = os.path.join(output_dir, 'data_exploration_quickstart.ipynb')
        with open(notebook_path, 'w') as f:
            f.write(notebook_content)
        
        self.logger.info(f"Sample notebook created: {notebook_path}")

def main():
    """Main execution function."""
    cleaner = DataCleaner()
    
    # Input and output paths
    input_path = "data/processed/final_comprehensive_dataset.csv"
    output_path = "data/processed/ml_ready_dataset.csv"
    
    print("🧹 === DATA CLEANING AND FORMATTING ===")
    print()
    
    # Load and clean dataset
    df = cleaner.load_and_clean_dataset(input_path)
    
    # Handle missing values
    df_clean = cleaner.handle_missing_values(df, method='smart')
    
    # Format data
    df_formatted = cleaner.add_feature_metadata(df_clean)
    
    # Create feature dictionary
    feature_dict = cleaner.create_feature_dictionary(df_formatted)
    
    # Save cleaned dataset
    output_file = cleaner.save_cleaned_dataset(df_formatted, output_path, feature_dict)
    
    # Create sample notebook
    cleaner.create_sample_notebook(df_formatted, "notebooks/")
    
    print()
    print("✅ DATA CLEANING COMPLETE!")
    print(f"📁 Clean dataset: {output_file}")
    print(f"📋 Feature dictionary: {output_file.replace('.csv', '_features.json')}")
    print(f"📓 Sample notebook: notebooks/data_exploration_quickstart.ipynb")
    print()
    print("🚀 Your team can now use the ML-ready dataset!")

if __name__ == "__main__":
    main()