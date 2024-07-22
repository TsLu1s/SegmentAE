from itertools import product
import pandas as pd
from segmentae.anomaly_detection import (SegmentAE, 
                                         Clustering)
from typing import List, Union

class SegmentAE_Optimizer:
    """
    An optimization pipeline for conducting anomaly detection experiments using diverse autoencoder architectures
    and clustering methodologies. This class facilitates the systematic exploration and rigorous evaluation
    of various combinations of autoencoders, clustering algorithms, and hyperparameters, aiming to identify
    the optimal configuration based on a specified performance metric.

    Attributes:
        n_clusters_list (List[int]): A list containing the numbers of clusters to explore.
        cluster_models (List[str]): A list of clustering algorithm names to be employed.
        autoencoder_models (List[Type[SegmentAE]]): A list of autoencoder model instances for evaluation.
        threshold_ratios (List[float]): A list of threshold ratios for anomaly detection.
        performance_metric (str): The metric used to evaluate and compare the performance of the models.
        optimal_segmentae (SegmentAE): The SegmentAE model that implements the highest performance AD configuration.
        best_threshold_ratio (float): The threshold ratio that achieves the optimal performance.
        best_n_clusters (int): The number of clusters that achieves the optimal performance.
        best_performance (float): The highest performance score achieved during the evaluation process.
    """

    def __init__(self, 
                 n_clusters_list: List[int] = [1, 2, 3, 4],
                 cluster_models: List[str] = ["KMeans", "MiniBatchKMeans", "GMM"],
                 autoencoder_models: list = [],
                 threshold_ratios: List[float] = [0.75, 1, 1.5, 2, 3, 4],
                 performance_metric: str = 'f1_score') -> None:
        """
        Initialize the AnomalyDetectionPipeline with the given parameters.

        Args:
            n_clusters_list (list): List of numbers of clusters to try.
            cluster_models (list): List of clustering algorithm names to use.
            autoencoder_models (list): List of autoencoder model instances to evaluate.
            threshold_ratios (list): List of threshold ratios to try for anomaly detection.
            performance_metric (str): Metric used to evaluate and compare model performances.
        """
        self.n_clusters_list = n_clusters_list
        self.cluster_models = cluster_models
        self.autoencoder_models = autoencoder_models
        self.threshold_ratios = threshold_ratios
        self.performance_metric = performance_metric
        self.optimal_segmentae = None  # New attribute to store the best SegmentAE model

        self.optimal_segmentae = None
        self.best_threshold_ratio: Union[float, None] = None
        self.best_n_clusters: Union[int, None] = None
        self.best_performance: float = float('-inf')

    def _update_optimal_model(self, current_model, current_performance: float, current_threshold: float, current_n_clusters: int) -> None:
        """
        Update the optimal model if the current model outperforms the previously recorded best model.

        This method compares the current model's performance with the best performance
        seen so far and updates the best model information if the current model performs better.

        Args:
            current_model (SegmentAE): The current SegmentAE model being evaluated.
            current_performance (float): The performance score of the current model.
            current_threshold (float): The threshold ratio used for the current model.
            current_n_clusters (int): The number of clusters used for the current model.
        """
        if current_performance > self.best_performance:
            self.optimal_segmentae = current_model
            self.best_performance = current_performance
            self.best_threshold_ratio = current_threshold
            self.best_n_clusters = current_n_clusters

    def optimize(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Execute the optimization pipeline to identify the optimal anomaly detection model.

        This method conducts a comprehensive evaluation of various combinations of autoencoders,
        clustering algorithms, numbers of clusters, and threshold ratios. Each combination is evaluated
        on the test dataset, and the optimal configuration is identified based on the specified performance metric.

        Args:
            X_train (pd.DataFrame): Training data features.
            X_test (pd.DataFrame): Test data features.
            y_test (pd.Series): Test data labels.

        Returns:
            SegmentAE: The SegmentAE model that achieved the highest performance.
        """
        
        metrics_log = []
        n = 1
        
        for n_clusters, cluster, autoencoder in product(
            self.n_clusters_list,
            self.cluster_models,
            self.autoencoder_models
        ):
            print(f"Iteration {n}")
            print(f"Cluster Model: {cluster}")
            print(f"Number of Clusters: {n_clusters}")
            print(f"Autoencoder: {type(autoencoder).__name__}")
            print("")

            # Clustering Implementation
            cl_model = Clustering(cluster_model=[cluster], n_clusters=n_clusters)
            cl_model.clustering_fit(X=X_train)

            # Autoencoder + Clustering Integration
            _sg = SegmentAE(ae_model=autoencoder, cl_model=cl_model)

            # Train Reconstruction
            _sg.reconstruction(input_data=X_train, threshold_metric='mse')

            # Multiple Threshold Ratio Reconstruction Evaluation
            for threshold_ratio in self.threshold_ratios:
                evaluation_results = _sg.evaluation(input_data=X_test, target_col=y_test, threshold_ratio=threshold_ratio)
                global_metrics = evaluation_results["global metrics"]
                
                current_performance = global_metrics[self.performance_metric].iloc[0]
                self._update_optimal_model(_sg, current_performance, threshold_ratio, n_clusters)

                global_metrics["Autoencoder"] = type(autoencoder).__name__
                global_metrics["Cluster"] = cluster
                global_metrics["N_Clusters"] = n_clusters

                metrics_log.append(global_metrics)

            n+=1

        self.leaderboard = pd.concat(metrics_log, axis=0).sort_values(self.performance_metric, ascending=False)

        print(f"\nBest Model Performance ({self.performance_metric}): {round(self.best_performance,6)}")
        if len(self.autoencoder_models)>1:
            print(f"Best Model Type: {type(autoencoder).__name__}")
        if len(self.cluster_models)>1:
            print(f"Best Number of Clusters: {self.best_n_clusters}")
        if len(self.threshold_ratios)>1:
            print(f"Best Threshold Ratio: {self.best_threshold_ratio}")
        
        return self.optimal_segmentae