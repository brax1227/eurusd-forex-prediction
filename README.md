# EUR/USD Forex Prediction with Machine Learning

A machine learning project for predicting EUR/USD forex movements using price data and financial news sentiment analysis.

## Quick Start

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/brax1227/eurusd-forex-prediction.git
cd eurusd-forex-prediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Get the Data
```bash
# Run the data collection pipeline (takes 5-10 minutes)
python scripts/get_data.py
```

This will create `data/processed/ml_ready_dataset.csv` with 1,781 records and 40 features.

### 3. Start Building Models
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/processed/ml_ready_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

# Basic info
print(f"Dataset shape: {df.shape}")
print(f"Target distribution: {df['target'].value_counts()}")

# Feature categories
price_features = ['open', 'high', 'low', 'close', 'volume']
technical_features = [col for col in df.columns if any(x in col for x in ['ma_', 'rsi', 'volatility', 'price_change'])]
sentiment_features = [col for col in df.columns if any(x in col for x in ['vader', 'textblob', 'news'])]

# All features for ML (38 total)
X_features = price_features + technical_features + sentiment_features
y_target = 'target'  # Binary: 0=down, 1=up next day
```

## Dataset Overview

**What you get:**
- **1,781 trading days** of EUR/USD data (2019-2025)
- **40 features**: 5 price + 13 technical indicators + 16 sentiment + 1 target
- **88.6% news coverage** from 11+ financial sources
- **Balanced target**: 50.9% vs 49.1% (nearly perfect)
- **High quality**: 99.7% complete data

**Features included:**
- **Price**: OHLCV data
- **Technical**: Moving averages, RSI, volatility measures  
- **Sentiment**: VADER and TextBlob scores from financial news
- **Target**: Next-day price direction (binary classification)

## Recommended Workflow

### Data Splitting
```python
# Use chronological splits for time series
train_end = '2023-12-31'
val_end = '2024-06-30'

train_data = df[df['date'] <= train_end]
val_data = df[(df['date'] > train_end) & (df['date'] <= val_end)]
test_data = df[df['date'] > val_end]

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
```

### Model Development
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Prepare features
X_train = train_data[X_features]
y_train = train_data[y_target]

# Handle missing values (see feature documentation)
X_train = X_train.fillna(method='ffill').fillna(0)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train baseline model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
X_val = val_data[X_features].fillna(method='ffill').fillna(0)
X_val_scaled = scaler.transform(X_val)
y_val = val_data[y_target]

y_pred = model.predict(X_val_scaled)
print(classification_report(y_val, y_pred))
```

## Project Structure

```
eurusd-forex-prediction/
├── data/
│   ├── processed/
│   │   ├── ml_ready_dataset.csv          # Main dataset for ML
│   │   └── ml_ready_dataset_features.json # Feature documentation
│   └── raw/                              # Raw data (auto-generated)
├── src/
│   └── data_pipeline/                    # Data collection modules
├── scripts/
│   └── get_data.py                       # Data collection script
├── config/
│   └── config.yaml                       # Configuration
├── notebooks/                            # Jupyter notebooks (create your own)
├── requirements.txt                      # Dependencies
└── README.md                             # This file
```

## Feature Documentation

**Price Features (5):**
- `open`, `high`, `low`, `close`: Daily OHLC prices
- `volume`: Trading volume (always 0 for forex)

**Technical Indicators (13):**
- `price_change`: Daily percentage change
- `ma_5`, `ma_10`, `ma_20`, `ma_50`: Moving averages
- `close_ma_X_ratio`: Price vs moving average ratios
- `volatility_5d`, `volatility_20d`: Rolling volatility
- `rsi`: Relative Strength Index
- `high_low_range`, `open_close_range`: Intraday ranges

**Sentiment Features (16):**
- `news_count`: Number of news articles per day
- `vader_compound_mean/std/min/max`: VADER sentiment scores
- `vader_positive/negative/neutral_mean/std`: Component scores
- `textblob_polarity/subjectivity_mean/std/min/max`: TextBlob scores

**Target Variable (1):**
- `target`: Binary (0=price down, 1=price up next day)

## Data Quality

- **Completeness**: 99.7% (only 202 missing values out of 71,240)
- **Coverage**: 88.6% of trading days have sentiment data
- **Balance**: 96.4% ratio between classes (907 vs 874)
- **Timespan**: 6.8 years of data (2019-2025)
- **Sources**: 11+ financial news feeds + market events

## Performance Targets

- **Baseline**: >55% accuracy (beat random)
- **Goal**: >65% accuracy with good F1-score
- **Focus**: Sentiment features should improve performance vs price-only models

## Team Workflow

### Recommended Approach
1. **Start with baseline models**: Random Forest, XGBoost
2. **Add advanced models**: LSTM, ensemble methods
3. **Feature engineering**: Create interaction features, lags
4. **Hyperparameter tuning**: Systematic optimization
5. **Evaluation**: Use F1-score, precision, recall

### Branch Strategy
```bash
# Create feature branch for your work
git checkout -b feature/your-name-model-type
# Example: git checkout -b feature/sean-random-forest

# Make changes and commit
git add .
git commit -m "Implement Random Forest baseline model"

# Push and create pull request
git push origin feature/your-name-model-type
```

## Troubleshooting

**If data collection fails:**
- Check internet connection (needs to fetch financial data)
- Some news sources may be temporarily unavailable
- The script will continue with available sources

**Missing values:**
- Use `df.fillna(method='ffill')` for technical indicators
- Use `df.fillna(0)` for sentiment features
- Or drop first 50 rows for complete data

**Memory issues:**
- Dataset is only 0.6MB, should work on any system
- If needed, sample the data: `df.sample(frac=0.8)`

**Ready to build some models? Start with `python scripts/get_data.py`**