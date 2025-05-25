import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def get_global_stats():
    """
    Get global statistics for real-time prediction
    These would typically come from a database or cache in production
    """
    return {
        'CUST_AVG_AMOUNT': 500.0,
        'CUST_STD_AMOUNT': 300.0,
        'CUST_MAX_AMOUNT': 2000.0,
        'CUST_MIN_AMOUNT': 10.0,
        'HOUR_AVG_AMOUNT': 450.0,
        'HOUR_STD_AMOUNT': 250.0,
        'amount_bins': np.linspace(0, 2000, 11),  # Same bins as training
        'amount_categories': [0, 100, 500, 1000, 2000, float('inf')]  # Same categories as training
    }

def create_amount_features(df, is_training=True, customer_stats=None, time_stats=None):
    """
    Create amount-based features consistently for both training and prediction
    """
    if is_training:
        # Calculate customer statistics
        customer_stats = df.groupby('CUSTOMER_ID').agg({
            'TX_AMOUNT': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        customer_stats.columns = ['CUSTOMER_ID', 'CUST_AVG_AMOUNT', 'CUST_STD_AMOUNT', 
                                'CUST_MIN_AMOUNT', 'CUST_MAX_AMOUNT', 'CUST_TX_COUNT']
        
        # Calculate time statistics
        time_stats = df.groupby('hour')['TX_AMOUNT'].agg(['mean', 'std']).reset_index()
        time_stats.columns = ['hour', 'HOUR_AVG_AMOUNT', 'HOUR_STD_AMOUNT']
        
        # Merge statistics
        df = df.merge(customer_stats, on='CUSTOMER_ID', how='left')
        df = df.merge(time_stats, on='hour', how='left')
    else:
        # Use provided statistics
        df = df.merge(customer_stats, on='CUSTOMER_ID', how='left')
        df = df.merge(time_stats, on='hour', how='left')
    
    # Create amount-based features
    df['AMOUNT_TO_CUST_AVG'] = df['TX_AMOUNT'] / df['CUST_AVG_AMOUNT']
    df['AMOUNT_TO_CUST_MAX'] = df['TX_AMOUNT'] / df['CUST_MAX_AMOUNT']
    df['AMOUNT_Z_SCORE'] = (df['TX_AMOUNT'] - df['CUST_AVG_AMOUNT']) / df['CUST_STD_AMOUNT']
    
    # Amount pattern features
    df['IS_AMOUNT_UNUSUAL'] = (abs(df['AMOUNT_Z_SCORE']) > 3).astype(int)
    df['IS_AMOUNT_EXTREME'] = (abs(df['AMOUNT_Z_SCORE']) > 5).astype(int)
    
    # Amount time patterns
    df['AMOUNT_TIME_Z_SCORE'] = (df['TX_AMOUNT'] - df['HOUR_AVG_AMOUNT']) / df['HOUR_STD_AMOUNT']
    
    return df, customer_stats, time_stats

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset with sophisticated amount-based features
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Convert TX_DATETIME to datetime
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    
    # Extract datetime features
    df['hour'] = df['TX_DATETIME'].dt.hour
    df['day_of_week'] = df['TX_DATETIME'].dt.dayofweek
    df['month'] = df['TX_DATETIME'].dt.month
    
    print("Creating amount-based features...")
    
    # Create amount features
    df, customer_stats, time_stats = create_amount_features(df, is_training=True)
    
    # Create amount bins and categories
    df['AMOUNT_BIN'] = pd.qcut(df['TX_AMOUNT'], q=10, labels=False)
    df['AMOUNT_CATEGORY'] = pd.cut(df['TX_AMOUNT'], 
                                  bins=[0, 100, 500, 1000, 2000, float('inf')],
                                  labels=False)
    
    # Drop unnecessary columns
    df = df.drop(['TX_DATETIME', 'Unnamed: 0'], axis=1)
    
    return df

def prepare_transaction_data(transaction_data, customer_stats=None, time_stats=None):
    """
    Prepare a single transaction for prediction with sophisticated amount features
    """
    # Convert transaction data to DataFrame
    df = pd.DataFrame([transaction_data])
    
    # Convert TX_DATETIME to datetime
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    
    # Extract datetime features
    df['hour'] = df['TX_DATETIME'].dt.hour
    df['day_of_week'] = df['TX_DATETIME'].dt.dayofweek
    df['month'] = df['TX_DATETIME'].dt.month
    
    # Create amount features
    df, _, _ = create_amount_features(df, is_training=False, 
                                    customer_stats=customer_stats,
                                    time_stats=time_stats)
    
    # Create amount bins and categories using global statistics
    global_stats = get_global_stats()
    df['AMOUNT_BIN'] = pd.cut(df['TX_AMOUNT'], bins=global_stats['amount_bins'], labels=False)
    df['AMOUNT_CATEGORY'] = pd.cut(df['TX_AMOUNT'], 
                                  bins=global_stats['amount_categories'],
                                  labels=False)
    
    # Drop the datetime column
    df = df.drop(['TX_DATETIME'], axis=1)
    
    return df