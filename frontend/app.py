import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import datetime, time
import plotly.express as px
import plotly.graph_objects as go
from backend import load_model, predict_fraud

# Set page config
st.set_page_config(
    page_title="Real-Time Transaction Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

def main():
    # Title and description
    st.title("💳 Real-Time Transaction Fraud Detection System")
    st.markdown("""
    This application uses machine learning to detect fraudulent transactions in real-time.
    Enter the transaction details below to get a prediction.
    """)
    
    # Load the model
    try:
        model = load_model()
        st.success("Machine Learning Model loaded successfully!")
    except FileNotFoundError as e:
        st.error(str(e))
        return
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        # Input fields
        transaction_id = st.number_input("Transaction ID", min_value=1, value=1234567)
        customer_id = st.number_input("Customer ID", min_value=1, value=4064)
        terminal_id = st.number_input("Terminal ID", min_value=1, value=75)
        tx_amount = st.number_input("Transaction Amount", min_value=0.0, value=619.72, format="%.2f")
        
    with col2:
        st.subheader("Time Information")
        # Date input
        tx_date = st.date_input("Transaction Date", value=datetime.now().date())
        
        # Time input using hour and minute selectors
        time_col1, time_col2 = st.columns(2)
        with time_col1:
            hour = st.selectbox("Hour", range(24), index=datetime.now().hour)
        with time_col2:
            minute = st.selectbox("Minute", range(60), index=datetime.now().minute)
        
        # Create time object
        tx_time = time(hour, minute)
        
        tx_time_seconds = st.number_input("Transaction Time (Seconds)", min_value=0, value=3600)
        tx_time_days = st.number_input("Transaction Time (Days)", min_value=0, value=137)
    
    # Combine date and time
    tx_datetime = datetime.combine(tx_date, tx_time)
    
    # Create transaction dictionary
    transaction = {
        'TRANSACTION_ID': int(transaction_id),
        'CUSTOMER_ID': int(customer_id),
        'TERMINAL_ID': int(terminal_id),
        'TX_AMOUNT': float(tx_amount),
        'TX_TIME_SECONDS': int(tx_time_seconds),
        'TX_TIME_DAYS': int(tx_time_days),
        'TX_DATETIME': tx_datetime.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Prediction button
    if st.button("Predict Fraud", type="primary"):
        with st.spinner("Analyzing transaction..."):
            # Make prediction
            prediction = predict_fraud(model, transaction)
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Create columns for results
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                # Fraud status with color
                if prediction['is_fraud']:
                    st.error("🚨 FRAUDULENT TRANSACTION DETECTED!")
                else:
                    st.success("✅ LEGITIMATE TRANSACTION")
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction['fraud_probability'] * 100,
                    title={'text': "Fraud Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with res_col2:
                # Transaction details
                st.markdown("### Transaction Details")
                st.json(transaction)
                
                # Features used
                #st.markdown("### Features Used")
                #st.write(", ".join(prediction['features_used']))

if __name__ == "__main__":
    main() 