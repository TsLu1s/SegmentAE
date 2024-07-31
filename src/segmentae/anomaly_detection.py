import warnings
warnings.filterwarnings("ignore", category=Warning)
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, 
                             mean_squared_error,
                             mean_absolute_error,
                             max_error)
from segmentae.metrics.performance_metrics import metrics_classification
from segmentae.preprocessing.core import Preprocessing
from segmentae.autoencoders.dense import DenseAutoencoder
from segmentae.autoencoders.batch_norm import BatchNormAutoencoder
from segmentae.autoencoders.ensemble import EnsembleAutoencoder
from segmentae.clusters.clustering_ensembling import Clustering
from typing import Optional

class SegmentAE:
    def __init__(self, 
                 ae_model,
                 cl_model):
        """
        SegmentAE class is designed to integrate an autoencoder model (ae_model) with a clustering model (cl_model) 
        to detect anomalies. This class leverages the autoencoder's ability to reconstruct input data and 
        the clustering model's capability to partition data into distinct groups.

        Parameters:
        - ae_model: An autoencoder model used for data reconstruction.
        - cl_model: A clustering model used for partitioning the data.

        Attributes:
        - preds_train, preds_test: Dictionaries to store predictions for training and testing phases.
        - _phase: String indicating the current phase of the process ("evaluation", "testing", "prediction").
        - threshold: Value used as the cutoff for anomaly detection.
        - _threshold_metric: The metric used to determine the threshold for anomaly detection.
        - reconstruction_eval, reconstruction_test, reconstruction_pred: Dictionaries to store reconstruction errors for evaluation, testing, and prediction phases.
        - results: Dictionary to store the final evaluation results.
        
        Note:
        - Normal Class Label: 0
        - Anomaly Class Label: 1
        """
        self.ae_model = ae_model
        self.cl_model = cl_model
        self.preds_train, self.preds_test = {}, {}
        self._phase = "evaluation"
        self.threshold = None
        self._threshold_metric = None
        self.reconstruction_eval = {}
        self.reconstruction_test = {}
        self.reconstruction_pred = {}
        self.results = {}

    def reconstruction(self,
                       input_data : pd.DataFrame,
                       target_col : Optional[pd.DataFrame] = None,
                       threshold_metric : str = "mse"): #  Options | mse, mae, rmse, max_error
        """
        Reconstructs input data using the autoencoder model and computes reconstruction errors for each cluster.

        Parameters:
        - input_data (pd.DataFrame): The input data to be reconstructed.
        - target_col (Optional[pd.DataFrame]): The target column containing ground truth labels.
        - threshold_metric (str): The metric used for computing the reconstruction threshold. Options are 'mse', 'mae', 'rmse', 'max_error'.

        Returns:
        - Depending on the phase, returns different attributes.
        """
        if threshold_metric == "mse": 
            self._threshold_metric = "MSE_Recons_error"
        elif threshold_metric == "mae": 
            self._threshold_metric = "MAE_Recons_error"
        elif threshold_metric == "rmse": 
            self._threshold_metric = "RMSE_Recons_error"
        elif threshold_metric == "max_error": 
            self._threshold_metric = "MaxError_Recons_error"
        
        clustering_detect = self.cl_model.cluster_prediction(X = input_data)
        X_train_, preds = input_data.copy(), {}
        
        for cluster in clustering_detect[self.cl_model.cluster_model[0]].unique(): # Iterate through clusters
            index = clustering_detect.index[clustering_detect[self.cl_model.cluster_model[0]] == cluster].tolist() # Get indices of data points in each cluster
            X_ = X_train_.loc[index] # Extract metadata for each cluster
            if target_col is not None: y_true = target_col.loc[index] 
            else: y_true = None # Extract metadata for each cluster
            predictions = pd.DataFrame(self.ae_model.predict(X_), columns = input_data.columns).astype(float) # Reconstruct features using autoencoder
            preds[cluster] = {"cluster":cluster, "real":X_, "y_true": y_true, "predictions": predictions, "indexs":index} # Store predictions for each cluster
    
        ae_clust_metrics, clust_preds = {}, preds.copy()
        for partition in range(len(clust_preds)): # Iterate through partitions
            real_values = clust_preds[partition]['real'].values # Get real values
            predicted_values = clust_preds[partition]['predictions'].values # Get predicted values
            index = clust_preds[partition]['indexs'] # Get indices of data points
            clust_indice = clust_preds[partition]['cluster'] # Get cluster indice
            
            mse_per_row, mae_per_row, rmse_per_row, max_error_per_row = [], [], [], []
            
            for i in range(len(real_values)): # Iterate through data points
                # Retrieve the real data point and its corresponding prediction
                row = real_values[i]
                pred_row = predicted_values[i]
                
                # Calculate the MSE, MAE, RMSE & Max_Error for the current data point
                mse = mean_squared_error(row, pred_row)
                mae = mean_absolute_error(row, pred_row) 
                rmse = np.sqrt(mse)
                max_err = max_error(row, pred_row)
                
                # Store the errors for the current data point
                mse_per_row.append(mse)
                mae_per_row.append(mae)
                rmse_per_row.append(rmse)
                max_error_per_row.append(max_err)
                
            metrics_df = pd.DataFrame({'MSE_Recons_error': mse_per_row,
                                       'MAE_Recons_error': mae_per_row,
                                       'RMSE_Recons_error': rmse_per_row,
                                       'Max_Recons_error': max_error_per_row,
                                       'Score': np.array(mse_per_row).mean() + np.array(mse_per_row).std()
                                       })
            
            col_metrics = pd.DataFrame({'Column': list(clust_preds[partition]['real'].columns),
                                        'MSE': list(np.mean(np.square(real_values - predicted_values), axis=0)), 
                                        'MAE': list(np.mean(np.abs(real_values - predicted_values), axis=0)), 
                                        'RMSE': np.sqrt(np.mean(np.square(real_values - predicted_values), axis=0)),
                                        'Max_Error': np.max(np.abs(real_values - predicted_values), axis=0),
                                        'partition': partition
                                        })
            
            
            total_metrics = pd.DataFrame({'Column': ['Total Metrics'],
                                          'MSE': [col_metrics['MSE'].mean()],
                                          'MAE': [col_metrics['MAE'].mean()],
                                          'RMSE': [col_metrics['RMSE'].mean()],
                                          'Max_Error': [col_metrics['Max_Error'].max()],
                                          })

            col_metrics = pd.concat([col_metrics, total_metrics], ignore_index=True)
            ae_clust_metrics[partition] = {"cluster":clust_indice, "metrics":metrics_df, "column_metrics": col_metrics, "indexs": index}
            
        if self._phase == "evaluation": 
            self.preds_train = preds
            self.reconstruction_eval = ae_clust_metrics
            
            return self
        
        elif self._phase == "testing":
            self.preds_test = preds
            self.reconstruction_test = ae_clust_metrics
            
            return self.preds_test, self.reconstruction_test
        
        elif self._phase == "prediction":
            self.preds_final = preds
            self.reconstruction_final = ae_clust_metrics
            
            return self.preds_final, self.reconstruction_final
        
    def evaluation(self,
                   input_data : pd.DataFrame,
                   target_col : pd.DataFrame,
                   threshold_ratio : float = 1.0):
        """
        Evaluates the model's performance by comparing reconstructed predicted anomalies against the true input data values.

        Parameters:
        - input_data (pd.DataFrame): The input data for evaluation.
        - target_col (pd.DataFrame): The ground truth labels.
        - threshold_ratio (float): The ratio used to adjust the reconstruction error threshold.

        Returns:
        - results (dict): A dictionary containing global and cluster-specific metrics, confusion matrices, and predictions.
        """
        self._phase = "testing"
        self.preds_test, self.reconstruction_test = self.reconstruction(input_data = input_data,
                                                                        target_col = target_col,
                                                                        threshold_metric = self._threshold_metric)
        cmts, fgms, tests = {}, [], []
        for cluster in self.reconstruction_eval.keys():
          if cluster not in self.reconstruction_test:
              print(f"Cluster {cluster} not found in Reconstruction")
              continue
            
            rec_error = self.reconstruction_eval[cluster]["metrics"][self._threshold_metric]
            self.threshold = np.mean(rec_error)*threshold_ratio

            print(f"Cluster {cluster} || Reconstruction Threshold : {round(self.threshold,5)}")
            if cluster == len(self.reconstruction_eval)-1: print("")
            metrics_test = self.reconstruction_test[cluster]["metrics"]
            
            predictions = self.preds_test[cluster]["predictions"]
            y_test = self.preds_test[cluster]["y_true"].reset_index(drop = True)
            index_t = self.preds_test[cluster]["indexs"]
            
            predictions['Predicted Anomalies'] = metrics_test[self._threshold_metric].apply(lambda x: 1 if x > self.threshold else 0)
            
            cm = confusion_matrix(y_test, predictions['Predicted Anomalies'])
            frag_metrics = metrics_classification(y_test, predictions['Predicted Anomalies'])
            frag_metrics["N_Cluster"] = cluster
            frag_metrics["Threshold Metric"] = self._threshold_metric.split('_')[0]
            frag_metrics["Threshold Value"] = round(self.threshold,6)
            fgms.append(frag_metrics)
            cmts[cluster] = {"CM_"+str(cluster): cm}
            
            ## Global Metrics
            test = pd.DataFrame({'index': index_t, 'y_test': y_test,'Predicted Anomalies': predictions['Predicted Anomalies']})
            tests.append(test)
        
        clusters_metrics = pd.concat(fgms, ignore_index=True)
        ytpred = pd.concat([df for df in tests], ignore_index=True)
        ytpred = ytpred.sort_values(by='index').set_index('index')
        global_metrics = metrics_classification(ytpred['y_test'], ytpred['Predicted Anomalies'])
        global_metrics["Threshold Metric"] = self._threshold_metric.split('_')[0]
        global_metrics["Threshold Ratio"] = threshold_ratio
         
        self.results = {"global metrics": global_metrics, 
                        "clusters metrics": clusters_metrics, 
                        "confusion matrix": cmts,
                        "y_true vs y_pred": ytpred}
        
        return self.results
    
    def detections(self,
                   input_data : pd.DataFrame,
                   threshold_ratio : float = 1.0):
        """
        Performs cluster optimized autoencoder anomaly detection predictions on new data.

        Parameters:
        - input_data (pd.DataFrame): New data for anomaly detection.
        - threshold_ratio (float): Ratio used to adjust the reconstruction error threshold.

        Returns:
        - anomaly_predictions (pd.DataFrame): DataFrame containing the predicted anomalies.
        """
        self._phase, tests = "prediction", []
        self.reconstruction(input_data = input_data,
                            target_col = None,
                            threshold_metric = self._threshold_metric)
        
        for cluster in self.reconstruction_eval.keys():
          if cluster not in self.reconstruction_final:
              continue
            rec_error = self.reconstruction_eval[cluster]["metrics"][self._threshold_metric]
            self.threshold = np.mean(rec_error)*threshold_ratio

            recons_metrics = self.reconstruction_final[cluster]["metrics"]

            predictions = self.preds_final[cluster]["predictions"]
            index_t = self.preds_final[cluster]["indexs"]
            
            predictions['Predicted Anomalies'] = recons_metrics[self._threshold_metric].apply(lambda x: 1 if x > self.threshold else 0)
            predictions['_index'] = index_t

            tests.append(predictions)
        
        anomaly_predictions = pd.concat([df for df in tests], ignore_index=True)
        anomaly_predictions = anomaly_predictions.sort_values(by='_index').reset_index(drop=True).drop('_index',axis=1)
        
        return anomaly_predictions

__all__ = [
    'metrics_classification',
    'DenseAutoencoder',
    'BatchNormAutoencoder',
    'EnsembleAutoencoder',
    'Clustering',
    'Preprocessing'
]

