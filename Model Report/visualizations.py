import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
import sys

def create_feature_importance_plot(importances, feature_names):
    """Create a horizontal bar plot of feature importances."""
    try:
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Analysis')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print("✓ Created feature importance plot")
    except Exception as e:
        print(f"Error creating feature importance plot: {str(e)}")

def create_performance_metrics_plot(metrics):
    """Create a radar plot of performance metrics using matplotlib."""
    try:
        # Number of variables
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the first value again to close the plot
        values += values[:1]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.4)
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Add title
        plt.title('Model Performance Metrics', size=15, y=1.1)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig('performance_metrics.png')
        plt.close()
        print("✓ Created performance metrics plot")
    except Exception as e:
        print(f"Error creating performance metrics plot: {str(e)}")

def create_cv_scores_plot(cv_scores):
    """Create a box plot of cross-validation scores."""
    try:
        plt.figure(figsize=(8, 6))
        plt.boxplot(cv_scores)
        plt.title('Cross-Validation Scores Distribution')
        plt.ylabel('F1 Score')
        plt.savefig('cv_scores.png')
        plt.close()
        print("✓ Created CV scores plot")
    except Exception as e:
        print(f"Error creating CV scores plot: {str(e)}")

def create_confusion_matrix_plot(y_true, y_pred):
    """Create an enhanced heatmap of the confusion matrix with additional metrics."""
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix', pad=20)
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Add metrics text
        metrics_text = f"""
        Accuracy: {accuracy:.2%}
        Precision: {precision:.2%}
        Recall: {recall:.2%}
        F1-Score: {f1:.2%}
        
        True Negatives: {tn}
        False Positives: {fp}
        False Negatives: {fn}
        True Positives: {tp}
        """
        
        # Plot metrics
        ax2.axis('off')
        ax2.text(0.1, 0.5, metrics_text, 
                fontsize=12, 
                verticalalignment='center',
                bbox=dict(facecolor='white', 
                         edgecolor='gray',
                         alpha=0.8,
                         boxstyle='round,pad=1'))
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created confusion matrix plot")
    except Exception as e:
        print(f"Error creating confusion matrix plot: {str(e)}")

def create_roc_curve_plot(y_true, y_pred_proba):
    """Create ROC curve plot."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()
        print("✓ Created ROC curve plot")
    except Exception as e:
        print(f"Error creating ROC curve plot: {str(e)}")

def create_learning_curve_plot(train_sizes, train_scores, test_scores):
    """Create a learning curve plot to show model performance vs training size."""
    try:
        plt.figure(figsize=(10, 6))
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, test_mean, label='Cross-validation score', color='red', marker='o')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')

        plt.title('Learning Curve')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created learning curve plot")
    except Exception as e:
        print(f"Error creating learning curve plot: {str(e)}")

def create_precision_recall_curve_plot(y_true, y_pred_proba):
    """Create precision-recall curve plot."""
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        average_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, 
                label=f'Precision-Recall curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created precision-recall curve plot")
    except Exception as e:
        print(f"Error creating precision-recall curve plot: {str(e)}")

def create_feature_correlation_plot(correlation_matrix):
    """Create a heatmap of feature correlations."""
    try:
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created feature correlation plot")
    except Exception as e:
        print(f"Error creating feature correlation plot: {str(e)}")

def create_prediction_distribution_plot(y_true, y_pred_proba):
    """Create a distribution plot of prediction probabilities."""
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot distribution for each class
        sns.kdeplot(data=y_pred_proba[y_true == 0], label='Class 0', shade=True)
        sns.kdeplot(data=y_pred_proba[y_true == 1], label='Class 1', shade=True)
        
        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title('Distribution of Prediction Probabilities')
        plt.legend()
        plt.grid(True)
        plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created prediction distribution plot")
    except Exception as e:
        print(f"Error creating prediction distribution plot: {str(e)}")

def create_calibration_curve_plot(y_true, y_pred_proba):
    """Create a calibration curve plot with improved visualization."""
    try:
        from sklearn.calibration import calibration_curve
        
        # Increase number of bins for smoother curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=20, strategy='quantile')
        
        plt.figure(figsize=(10, 6))
        
        # Plot calibration curve
        plt.plot(prob_pred, prob_true, marker='o', label='Model', color='blue', linewidth=2)
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated', color='red', linewidth=2)
        
        # Add confidence intervals
        from scipy import stats
        n_samples = len(y_true)
        confidence = 0.95
        z = stats.norm.ppf((1 + confidence) / 2)
        std = np.sqrt((prob_true * (1 - prob_true)) / n_samples)
        plt.fill_between(prob_pred, 
                        prob_true - z * std,
                        prob_true + z * std,
                        alpha=0.2,
                        color='blue',
                        label='95% Confidence Interval')
        
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('True Probability', fontsize=12)
        plt.title('Calibration Curve with Confidence Intervals', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add Brier score
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(y_true, y_pred_proba)
        plt.text(0.05, 0.95, f'Brier Score: {brier:.3f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('calibration_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created calibration curve plot")
    except Exception as e:
        print(f"Error creating calibration curve plot: {str(e)}")

if __name__ == "__main__":
    print("Starting visualization generation...")
    
    # Example usage
    # These would be replaced with actual data from your model
    feature_importances = np.array([0.2615, 0.2465, 0.1628, 0.0904, 0.0585])
    feature_names = ['AMOUNT_TIME_Z_SCORE', 'TX_AMOUNT', 'AMOUNT_CATEGORY', 
                    'AMOUNT_BIN', 'CUST_AVG_AMOUNT']
    
    metrics = {
        'accuracy': 0.9833,
        'precision': 1.0,
        'recall': 0.9667,
        'f1': 0.9828
    }
    
    cv_scores = [0.9830, 0.9826, 0.9829, 0.9826, 0.9827]
    
    # Generate more realistic example data for better calibration demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Generate true labels with some class imbalance
    y_true = np.random.binomial(1, 0.3, n_samples)
    
    # Generate predicted probabilities that are well-calibrated
    y_pred_proba = np.zeros(n_samples)
    for i in range(n_samples):
        if y_true[i] == 1:
            y_pred_proba[i] = np.random.beta(8, 2)  # Higher probabilities for positive class
        else:
            y_pred_proba[i] = np.random.beta(2, 8)  # Lower probabilities for negative class
    
    # Generate predictions based on 0.5 threshold
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Example data for learning curve
    train_sizes = np.array([100, 200, 300, 400, 500])
    train_scores = np.array([[0.95, 0.96, 0.94], [0.96, 0.97, 0.95], 
                           [0.97, 0.98, 0.96], [0.98, 0.99, 0.97], 
                           [0.99, 0.99, 0.98]])
    test_scores = np.array([[0.93, 0.94, 0.92], [0.94, 0.95, 0.93], 
                          [0.95, 0.96, 0.94], [0.96, 0.97, 0.95], 
                          [0.97, 0.98, 0.96]])
    
    # Example data for feature correlation
    correlation_matrix = np.array([
        [1.0, 0.3, -0.2, 0.1, 0.4],
        [0.3, 1.0, 0.1, -0.3, 0.2],
        [-0.2, 0.1, 1.0, 0.5, -0.1],
        [0.1, -0.3, 0.5, 1.0, 0.2],
        [0.4, 0.2, -0.1, 0.2, 1.0]
    ])
    
    # Create visualizations
    create_feature_importance_plot(feature_importances, feature_names)
    create_performance_metrics_plot(metrics)
    create_cv_scores_plot(cv_scores)
    create_confusion_matrix_plot(y_true, y_pred)
    create_roc_curve_plot(y_true, y_pred_proba)
    create_learning_curve_plot(train_sizes, train_scores, test_scores)
    create_precision_recall_curve_plot(y_true, y_pred_proba)
    create_feature_correlation_plot(correlation_matrix)
    create_prediction_distribution_plot(y_true, y_pred_proba)
    create_calibration_curve_plot(y_true, y_pred_proba)
    
    print("\nVisualization generation completed!") 