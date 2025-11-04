"""
Data Preprocessing Module
Handles data cleaning, feature engineering, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import logging
import os
from typing import Tuple, Optional, Dict, Any
import joblib

class DataPreprocessor:
    """Handles all data preprocessing tasks for the forex prediction model."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.scaler = None
        self.feature_columns = None
        self.target_column = 'target'
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for the preprocessor."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            self.logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
            
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Handle missing values
        self.logger.info("Handling missing values...")
        
        # For price data, forward fill then backward fill
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        # For technical indicators, use forward fill
        tech_columns = [col for col in df_clean.columns if any(x in col for x in ['ma_', 'rsi', 'volatility'])]
        for col in tech_columns:
            df_clean[col] = df_clean[col].fillna(method='ffill')
        
        # For sentiment data, fill with neutral values
        sentiment_columns = [col for col in df_clean.columns if any(x in col for x in ['vader', 'textblob'])]
        for col in sentiment_columns:
            if 'count' in col:
                df_clean[col] = df_clean[col].fillna(0)
            elif 'neutral' in col:
                df_clean[col] = df_clean[col].fillna(1.0)
            else:
                df_clean[col] = df_clean[col].fillna(0.0)
        
        # Remove rows where target is missing
        if self.target_column in df_clean.columns:
            df_clean = df_clean.dropna(subset=[self.target_column])
        
        # Handle outliers using IQR method for price-based features
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        exclude_columns = ['target', 'date'] + [col for col in df_clean.columns if 'vader' in col or 'textblob' in col]
        
        for col in numeric_columns:
            if col not in exclude_columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        final_rows = len(df_clean)
        self.logger.info(f"Data cleaning completed: {initial_rows} -> {final_rows} rows")
        
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for better model performance.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            return df
            
        df_features = df.copy()
        
        self.logger.info("Engineering additional features...")
        
        # Lag features for sentiment
        sentiment_cols = [col for col in df_features.columns if any(x in col for x in ['vader_compound', 'textblob_polarity'])]
        for col in sentiment_cols:
            for lag in [1, 2, 3]:
                df_features[f"{col}_lag_{lag}"] = df_features[col].shift(lag)
        
        # Rolling sentiment features
        for col in sentiment_cols:
            df_features[f"{col}_rolling_3d"] = df_features[col].rolling(window=3).mean()
            df_features[f"{col}_rolling_7d"] = df_features[col].rolling(window=7).mean()
        
        # Price momentum features
        if 'close' in df_features.columns:
            for period in [3, 5, 10]:
                df_features[f'momentum_{period}d'] = df_features['close'] / df_features['close'].shift(period) - 1
                df_features[f'price_change_{period}d'] = df_features['close'].pct_change(periods=period)
        
        # Interaction features between price and sentiment
        if 'vader_compound_mean' in df_features.columns and 'price_change' in df_features.columns:
            df_features['sentiment_price_interaction'] = df_features['vader_compound_mean'] * df_features['price_change']
        
        # Day of week features (forex markets have weekly patterns)
        if 'date' in df_features.columns:
            df_features['day_of_week'] = df_features['date'].dt.dayofweek
            df_features['is_monday'] = (df_features['day_of_week'] == 0).astype(int)
            df_features['is_friday'] = (df_features['day_of_week'] == 4).astype(int)
        
        # Volatility regime features
        if 'volatility_20d' in df_features.columns:
            vol_median = df_features['volatility_20d'].median()
            df_features['high_volatility_regime'] = (df_features['volatility_20d'] > vol_median).astype(int)
        
        self.logger.info(f"Feature engineering completed: {len(df_features.columns)} total features")
        
        return df_features
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if df.empty:
            return pd.DataFrame(), pd.Series()
        
        # Define feature columns (exclude non-feature columns)
        exclude_columns = ['date', 'target', 'merge_date']
        if 'text_content' in df.columns:
            exclude_columns.append('text_content')
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        self.feature_columns = feature_columns
        
        # Prepare features
        X = df[feature_columns].copy()
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Prepare target
        if self.target_column in df.columns:
            y = df[self.target_column].copy()
        else:
            self.logger.warning(f"Target column '{self.target_column}' not found")
            y = pd.Series()
        
        self.logger.info(f"Prepared {len(feature_columns)} features and {len(y)} target values")
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
                      scaler_type: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using specified scaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple of scaled (X_train, X_test)
        """
        if X_train.empty:
            return X_train, X_test or pd.DataFrame()
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.logger.warning(f"Unknown scaler type: {scaler_type}. Using StandardScaler.")
            self.scaler = StandardScaler()
        
        # Fit and transform training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data if provided
        if X_test is not None and not X_test.empty:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        else:
            X_test_scaled = pd.DataFrame()
        
        self.logger.info(f"Features scaled using {scaler_type} scaler")
        
        return X_train_scaled, X_test_scaled
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, 
                              method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance in the dataset.
        
        Args:
            X: Features
            y: Target
            method: Resampling method ('smote', 'undersample', 'oversample', 'smoteenn')
            
        Returns:
            Tuple of resampled (X, y)
        """
        if X.empty or y.empty:
            return X, y
        
        # Check class distribution
        class_counts = y.value_counts()
        self.logger.info(f"Original class distribution: {dict(class_counts)}")
        
        if len(class_counts) < 2:
            self.logger.warning("Only one class present, skipping resampling")
            return X, y
        
        # Apply resampling
        try:
            if method == 'smote':
                resampler = SMOTE(random_state=42)
            elif method == 'undersample':
                resampler = RandomUnderSampler(random_state=42)
            elif method == 'smoteenn':
                resampler = SMOTEENN(random_state=42)
            else:
                self.logger.warning(f"Unknown resampling method: {method}")
                return X, y
            
            X_resampled, y_resampled = resampler.fit_resample(X, y)
            
            # Convert back to pandas
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled)
            
            new_class_counts = y_resampled.value_counts()
            self.logger.info(f"Resampled class distribution: {dict(new_class_counts)}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.logger.error(f"Error in resampling: {e}")
            return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                  test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion for test set
            val_size: Proportion for validation set (from training data)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if X.empty or y.empty:
            return tuple([pd.DataFrame() if 'X' in str(i) else pd.Series() for i in range(6)])
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, filepath: str):
        """Save the preprocessor state."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            preprocessor_state = {
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column
            }
            joblib.dump(preprocessor_state, filepath)
            self.logger.info(f"Preprocessor saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving preprocessor: {e}")
    
    def load_preprocessor(self, filepath: str):
        """Load the preprocessor state."""
        try:
            preprocessor_state = joblib.load(filepath)
            self.scaler = preprocessor_state['scaler']
            self.feature_columns = preprocessor_state['feature_columns']
            self.target_column = preprocessor_state['target_column']
            self.logger.info(f"Preprocessor loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading preprocessor: {e}")

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    data_file = "data/processed/eurusd_training_dataset.csv"
    
    if os.path.exists(data_file):
        # Load data
        df = preprocessor.load_data(data_file)
        
        # Clean data
        df_clean = preprocessor.clean_data(df)
        
        # Engineer features
        df_features = preprocessor.engineer_features(df_clean)
        
        # Prepare features and target
        X, y = preprocessor.prepare_features_target(df_features)
        
        if not X.empty and not y.empty:
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
            
            # Scale features
            X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
            X_val_scaled, _ = preprocessor.scale_features(X_val, scaler_type='standard')
            
            # Handle class imbalance
            X_train_balanced, y_train_balanced = preprocessor.handle_class_imbalance(
                X_train_scaled, y_train, method='smote'
            )
            
            print(f"Preprocessing completed successfully!")
            print(f"Training set: {X_train_balanced.shape}")
            print(f"Validation set: {X_val_scaled.shape}")
            print(f"Test set: {X_test_scaled.shape}")
            
            # Save preprocessor
            preprocessor.save_preprocessor("experiments/preprocessor.pkl")
        else:
            print("Failed to prepare features and target")
    else:
        print(f"Data file not found: {data_file}")