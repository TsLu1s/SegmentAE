import pandas as pd
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             f1_score, 
                             recall_score,
                             mean_absolute_error,
                             mean_squared_error,
                             r2_score,
                             max_error)

def metrics_classification(y_true, y_pred):
    """
    Calculate various classification model evaluation metrics.

    Parameters:
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns:
    pandas.DataFrame
        DataFrame containing accuracy, precision, recall, and F1 score metrics.
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate precision
    precision = precision_score(y_true, y_pred)
    
    # Calculate recall
    recall = recall_score(y_true, y_pred)
    
    # Calculate f1
    f1 = f1_score(y_true, y_pred)
    
    # Create a dictionary to store the metrics
    metrics = {'Accuracy': accuracy,
               'Precision': precision,
               'Recall': recall,
               'F1 Score': f1}
    
    # Convert metrics dictionary to DataFrame
    metrics = pd.DataFrame(metrics, index=[0])
    
    return metrics

def metrics_regression(y_true, y_pred):
    """
    Calculate various regression model evaluation metrics.

    Parameters:
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns:
    pandas.DataFrame
        DataFrame containing Mean Absolute Error, Mean Squared Error, 
        Root Mean Squared Error, and R-squared metrics.
    """
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    
    # Calculate R-squared
    r2 = r2_score(y_true, y_pred)
    
    # Calculate Max Error
    maxerror = max_error(y_true, y_pred)
    
    # Create a dictionary to store the metrics
    metrics = {'Mean Absolute Error': mae,
               'Mean Squared Error': mse,
               'Root Mean Squared Error': rmse,
               'R-squared': r2,
               'Max Error': maxerror}
    
    # Convert metrics dictionary to DataFrame
    metrics = pd.DataFrame(metrics, index=[0])
    
    return metrics