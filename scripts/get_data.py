#!/usr/bin/env python3
"""
EUR/USD Dataset Collection Script
Collects price data and news sentiment to create ML-ready dataset.
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline.price_data_collector import PriceDataCollector
from data_pipeline.enhanced_news_collector import EnhancedNewsCollector
from data_pipeline.historical_news_collector import HistoricalNewsCollector
from data_pipeline.data_merger import DataMerger
from data_pipeline.data_cleaner import DataCleaner

def main():
    print("=== EUR/USD Dataset Collection Pipeline ===")
    print("Collecting comprehensive EUR/USD data for ML modeling...")
    print()
    
    # Configuration
    start_date = '2019-01-01'
    max_articles = 150
    output_dir = 'data/raw'
    
    # Initialize collectors
    price_collector = PriceDataCollector(data_dir=output_dir)
    news_collector = EnhancedNewsCollector(data_dir=output_dir)
    historical_collector = HistoricalNewsCollector(data_dir=output_dir)
    
    # Step 1: Collect EUR/USD price data
    print("1. Collecting EUR/USD price data...")
    price_data = price_collector.fetch_yahoo_data(start_date=start_date)
    
    if not price_data.empty:
        price_file = price_collector.save_price_data(price_data, "eurusd_prices.csv")
        print(f"   ✓ Saved {len(price_data):,} price records")
    else:
        print("   ✗ Failed to collect price data")
        return 1
    
    # Step 2: Collect current news data
    print("2. Collecting financial news...")
    current_news = news_collector.collect_comprehensive_news(max_articles_per_feed=max_articles)
    
    if not current_news.empty:
        current_sentiment = news_collector.process_sentiment_enhanced(current_news)
        current_file = news_collector.save_enhanced_data(current_sentiment, "current_news.csv")
        print(f"   ✓ Saved {len(current_sentiment):,} news articles")
    else:
        print("   ⚠ No current news collected")
        current_file = None
    
    # Step 3: Generate historical sentiment data
    print("3. Generating historical market sentiment...")
    historical_sentiment = historical_collector.create_comprehensive_historical_dataset(start_date)
    
    if not historical_sentiment.empty:
        historical_file = historical_collector.save_historical_data(historical_sentiment, "historical_sentiment.csv")
        print(f"   ✓ Generated {len(historical_sentiment):,} historical records")
    else:
        print("   ✗ Failed to generate historical sentiment")
        return 1
    
    # Step 4: Combine all news data
    print("4. Combining news sources...")
    all_news_data = []
    
    if current_file and os.path.exists(current_file):
        current_data = pd.read_csv(current_file)
        all_news_data.append(current_data)
        print(f"   ✓ Added {len(current_data):,} current articles")
    
    if historical_file and os.path.exists(historical_file):
        historical_data = pd.read_csv(historical_file)
        all_news_data.append(historical_data)
        print(f"   ✓ Added {len(historical_data):,} historical records")
    
    # Combine and save
    combined_news = pd.concat(all_news_data, ignore_index=True)
    combined_news['published_date'] = pd.to_datetime(combined_news['published_date'], utc=True, errors='coerce')
    combined_news = combined_news.sort_values('published_date')
    combined_news = combined_news.drop_duplicates(subset=['published_date', 'title'], keep='first')
    
    combined_file = os.path.join(output_dir, "all_news_data.csv")
    combined_news.to_csv(combined_file, index=False)
    print(f"   ✓ Combined dataset: {len(combined_news):,} total records")
    
    # Step 5: Create training dataset
    print("5. Creating ML training dataset...")
    merger = DataMerger(processed_data_dir="data/processed")
    
    training_file = merger.create_training_dataset(
        price_file,
        combined_file,
        "raw_dataset.csv"
    )
    
    if not training_file:
        print("   ✗ Failed to create training dataset")
        return 1
    
    # Step 6: Clean and format for ML
    print("6. Cleaning and formatting dataset...")
    cleaner = DataCleaner()
    
    # Load and clean
    raw_df = cleaner.load_and_clean_dataset(training_file)
    clean_df = cleaner.handle_missing_values(raw_df, method='smart')
    final_df = cleaner.add_feature_metadata(clean_df)
    
    # Create feature dictionary
    feature_dict = cleaner.create_feature_dictionary(final_df)
    
    # Save final dataset
    final_file = cleaner.save_cleaned_dataset(
        final_df, 
        "data/processed/ml_ready_dataset.csv", 
        feature_dict
    )
    
    if final_file:
        print("   ✓ ML-ready dataset created")
        
        # Final summary
        news_coverage = (final_df['news_count'] > 0).sum()
        print()
        print("=== DATASET READY FOR ML MODELING ===")
        print(f"📊 Records: {len(final_df):,}")
        print(f"📈 Features: {len(final_df.columns)}")
        print(f"📰 News coverage: {news_coverage:,}/{len(final_df):,} days ({news_coverage/len(final_df)*100:.1f}%)")
        print(f"🎯 Target balance: {final_df['target'].value_counts().to_dict()}")
        print()
        print("Files created:")
        print(f"  - {final_file}")
        print(f"  - {final_file.replace('.csv', '_features.json')}")
        print()
        print("Next steps:")
        print("  1. Load data: df = pd.read_csv('data/processed/ml_ready_dataset.csv')")
        print("  2. Start building ML models!")
        
    else:
        print("   ✗ Failed to create final dataset")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())