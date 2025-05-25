import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(n_samples=1000):
    """Generate sample transaction data for testing."""
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate transaction amounts
    amounts = np.random.lognormal(mean=4, sigma=1, size=n_samples)
    
    # Generate customer IDs
    customer_ids = np.random.randint(1, 100, size=n_samples)
    
    # Calculate customer average amounts
    customer_avg_amounts = {}
    for cust_id in np.unique(customer_ids):
        customer_avg_amounts[cust_id] = np.mean(amounts[customer_ids == cust_id])
    
    # Create features
    data = {
        'timestamp': dates,
        'customer_id': customer_ids,
        'TX_AMOUNT': amounts,
        'CUST_AVG_AMOUNT': [customer_avg_amounts[cust_id] for cust_id in customer_ids]
    }
    
    # Calculate time-based features
    df = pd.DataFrame(data)
    df['AMOUNT_TIME_Z_SCORE'] = (df['TX_AMOUNT'] - df['TX_AMOUNT'].rolling(window=24).mean()) / df['TX_AMOUNT'].rolling(window=24).std()
    
    # Create amount categories
    df['AMOUNT_CATEGORY'] = pd.qcut(df['TX_AMOUNT'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Create amount bins
    df['AMOUNT_BIN'] = pd.cut(df['TX_AMOUNT'], 
                             bins=[0, 100, 500, 1000, 5000, float('inf')], 
                             labels=['0-100', '100-500', '500-1000', '1000-5000', '5000+'])
    
    # Generate labels (fraud/no fraud)
    # Higher amounts and unusual patterns are more likely to be fraud
    fraud_prob = 1 / (1 + np.exp(-(df['AMOUNT_TIME_Z_SCORE'] + df['TX_AMOUNT']/1000)))
    df['is_fraud'] = np.random.binomial(1, fraud_prob)
    
    # Fill NaN values
    df = df.fillna(method='bfill')
    
    return df

if __name__ == "__main__":
    # Generate sample data
    sample_data = generate_sample_data()
    
    # Save to CSV
    sample_data.to_csv('Model Report/sample_transactions.csv', index=False)
    print("Sample data generated and saved to 'Model Report/sample_transactions.csv'")
    print(f"Total samples: {len(sample_data)}")
    print(f"Fraud rate: {sample_data['is_fraud'].mean():.2%}") 