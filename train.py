from backend import (
    load_and_preprocess_data,
    train_model,
    evaluate_model,
    save_model
)
from sklearn.model_selection import train_test_split
import joblib

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('dataset/cleaned_transactions.csv')
    
    # Display basic information about the dataset
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    # Assuming TX_FRAUD is the target variable
    X = df.drop(['TX_FRAUD', 'TX_FRAUD_SCENARIO'], axis=1)
    y = df['TX_FRAUD']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluation_results = evaluate_model(model, X_test, y_test)
    
    # Save the model
    save_model(model)
    
    # Save customer and time statistics
    print("\nSaving statistics...")
    customer_stats = df.groupby('CUSTOMER_ID').agg({
        'TX_AMOUNT': ['mean', 'std', 'min', 'max', 'count']
    }).reset_index()
    customer_stats.columns = ['CUSTOMER_ID', 'CUST_AVG_AMOUNT', 'CUST_STD_AMOUNT', 
                            'CUST_MIN_AMOUNT', 'CUST_MAX_AMOUNT', 'CUST_TX_COUNT']
    joblib.dump(customer_stats, 'customer_stats.joblib')
    
    time_stats = df.groupby('hour')['TX_AMOUNT'].agg(['mean', 'std']).reset_index()
    time_stats.columns = ['hour', 'HOUR_AVG_AMOUNT', 'HOUR_STD_AMOUNT']
    joblib.dump(time_stats, 'time_stats.joblib')
    print("Statistics saved successfully!")

if __name__ == "__main__":
    main() 