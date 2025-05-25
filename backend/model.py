import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

def train_model(X_train, y_train):
    """
    Train the Random Forest model with cross-validation and feature importance analysis
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    print("\nTraining Random Forest model...")
    
    # Initialize the model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Perform cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train the model on full training set
    rf_model.fit(X_train, y_train)
    
    # Analyze feature importance
    print("\nFeature Importance Analysis:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nTop 5 most important features:")
    print(feature_importance.head())
    
    return rf_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance
    """
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Print accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy Score: {accuracy:.4f}")
    
    return {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': cm,
        'accuracy': accuracy
    }

def save_model(model, filepath='fraud_detection_model.joblib'):
    """
    Save the trained model
    """
    print("\nSaving model...")
    joblib.dump(model, filepath)
    print(f"Model saved as '{filepath}'")

def load_model(filepath='fraud_detection_model.joblib'):
    """
    Load the trained model
    """
    try:
        return joblib.load(filepath)
    except:
        raise FileNotFoundError(f"Model file not found at {filepath}") 