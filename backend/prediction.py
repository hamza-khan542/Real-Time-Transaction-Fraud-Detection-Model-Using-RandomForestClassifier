from .preprocessing import prepare_transaction_data, get_global_stats
import joblib
import os

def load_customer_stats():
    """
    Load customer statistics from file
    In production, this would come from a database
    """
    try:
        return joblib.load('customer_stats.joblib')
    except:
        return None

def load_time_stats():
    """
    Load time statistics from file
    In production, this would come from a database
    """
    try:
        return joblib.load('time_stats.joblib')
    except:
        return None

def predict_fraud(model, transaction_data):
    """
    Predict fraud for a new transaction
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    transaction_data : dict
        Dictionary containing transaction details
    
    Returns:
    --------
    dict
        Dictionary containing prediction results
    """
    # Load statistics
    customer_stats = load_customer_stats()
    time_stats = load_time_stats()
    
    # Prepare the transaction data
    df = prepare_transaction_data(transaction_data, customer_stats, time_stats)
    
    # Get feature names
    feature_names = df.columns.tolist()
    
    # Make prediction
    fraud_probability = model.predict_proba(df)[0][1]
    is_fraud = model.predict(df)[0]
    
    return {
        'is_fraud': bool(is_fraud),
        'fraud_probability': float(fraud_probability),
        'features_used': feature_names
    } 