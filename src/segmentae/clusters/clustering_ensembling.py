import pandas as pd
from typing import List
from segmentae.clusters.clustering import (KMeans_Clustering,
                                           MiniBatchKMeans_Clustering,
                                           GaussianMixture_Clustering,
                                           Agglomerative_Clustering)

class Clustering:
    def __init__(self, 
                 cluster_model: List[str] = ['KMeans'],
                 n_clusters : int = 3,
                 random_state : int = 0, 
                 covariance_type : str = "full"):
        """
        The Clustering class is designed for implementing multiple clustering algorithms. It allows the user to specify
        a variety of clustering models, the number of clusters, and other relevant parameters to fit and predict cluster
        assignments for a given dataset.

        Parameters:
        - cluster_model (List[str]): A list of clustering models to be used. The default model is 'KMeans'. Other options include 'MiniBatchKMeans', 'GMM', and 'Agglomerative'.
        - n_clusters (int): The number of clusters to form. The default value is 3.
        - random_state (int): Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
        - covariance_type (str): String describing the type of covariance parameters to use for GMM. It must be one of 'full', 'tied', 'diag', or 'spherical'.

        Attributes:
        - cmodel (object): Placeholder for the clustering model that will be instantiated and fitted.
        - clustering_dict (dict): Dictionary to store fitted clustering models keyed by the model name.
        """
        self.cluster_model = cluster_model
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.cmodel = None # Placeholder for the clustering model
        self.clustering_dict = {} # Dictionary to store fitted clustering models
        
    def clustering_fit(self, X : pd.DataFrame):
        """
        Fits multiple clustering models to the provided dataset. This method performs the following steps:
    
        1. Initialization:
           - The method iterates over the list of specified clustering models (`self.cluster_model`). Each model in this list is instantiated and fitted to the dataset `X`.
           - The available models include KMeans, MiniBatchKMeans, Gaussian Mixture Models (GMM), and Agglomerative Clustering, each suitable for different types of clustering tasks and data characteristics.
    
        2. Model Fitting:
           - For each clustering model specified in the `self.cluster_model` list:
             - **KMeans**: The KMeans algorithm partitions the data into `n_clusters` clusters, where each cluster is represented by the mean of its points (centroid). The algorithm aims to minimize the variance within each cluster. The `random_state` parameter ensures reproducibility.
             - **MiniBatchKMeans**: This variant of KMeans uses mini-batches to reduce computational cost while approximating the results of standard KMeans. It is suitable for large datasets.
             - **Gaussian Mixture Model (GMM)**: GMM assumes that the data is generated from a mixture of several Gaussian distributions with unknown parameters. It uses the Expectation-Maximization (EM) algorithm to find the mixture model that best fits the data. The `covariance_type` parameter determines the shape of the covariance matrices.
             - **Agglomerative Clustering**: This hierarchical clustering method builds nested clusters by merging or splitting them successively. It does not require specifying the number of clusters in advance, making it useful for exploratory data analysis.
           - Each instantiated model is fitted to the dataset `X`. The `fit` method of each model is called with `X` as the input, which performs the actual clustering.
    
        3. Model Storage:
           - The fitted model is stored in the `self.clustering_dict` dictionary, which maps model names to their corresponding fitted model instances.
           - If the dictionary is empty, the first fitted model is directly added. For subsequent models, the dictionary is updated with the new model and its corresponding instance.
           - This approach ensures that all specified models are available for later use, such as making predictions or evaluating model performance.
    
        4. Return:
           - The method returns the instance of the `Clustering` class (`self`), allowing for method chaining and further operations on the fitted models.
    
        Parameters:
        - X (pd.DataFrame): A pandas DataFrame containing the dataset to be clustered. Each row represents a sample, and each column represents a feature.
    
        Returns:
        - self (Clustering): The instance of the Clustering class with fitted models stored in `self.clustering_dict`.
        """
        for model in self.cluster_model:
            
            if model == 'KMeans':
                self.cmodel = KMeans_Clustering(n_clusters = self.n_clusters,
                                                random_state = self.random_state)
                self.cmodel.fit(X = X)
    
            elif model == 'MiniBatchKMeans':
                self.cmodel = MiniBatchKMeans_Clustering(n_clusters = self.n_clusters,
                                                         random_state = self.random_state)
                self.cmodel.fit(X = X)
    
            elif model == 'GMM':
                self.cmodel = GaussianMixture_Clustering(n_components = self.n_clusters,
                                                         covariance_type = self.covariance_type)
                self.cmodel.fit(X = X)
            
            elif model == 'Agglomerative':
                self.cmodel = Agglomerative_Clustering(n_clusters = self.n_clusters)
                self.cmodel.fit(X = X)

            # Store the fitted model information in a dictionary
            if len(self.clustering_dict.keys()) == 0:
                self.clustering_dict = {model : self.cmodel}
            elif len(self.clustering_dict.keys())>0:
                cluster = {model: self.cmodel}
                self.clustering_dict.update(cluster)
        

        return self

    def cluster_prediction(self, X : pd.DataFrame):
        """
        Predicts cluster assignments for the provided dataset using the fitted clustering models.

        This method performs the following steps:

        1. Initialization:
           - An empty pandas DataFrame (`results`) is initialized to store the clustering predictions for each model.

        2. Prediction:
           - For each model stored in `self.clustering_dict`:
             - The model is retrieved from the dictionary.
             - The `predict` method of the model is called with `X` as input to obtain the cluster assignments.
             - The predictions are stored in a new column of the `results` DataFrame, where the column name corresponds to the model name.

        3. Return:
           - The method returns the `results` DataFrame, which contains the cluster assignments for each model. Each column in the DataFrame represents the predictions from a different clustering model.

        Parameters:
        - X (pd.DataFrame): A pandas DataFrame containing the dataset to be clustered. Each row represents a sample, and each column represents a feature.

        Returns:
        - results (pd.DataFrame): A pandas DataFrame containing the cluster assignments for each model. Each column corresponds to a different clustering model.
        """
        results = pd.DataFrame()

        # Iterate through fitted models
        for model in self.clustering_dict.keys(): 
            self.cmodel = self.clustering_dict[model]
            predictions = self.cmodel.predict(X = X)
            # Concatenate results to the output DataFrame
            results = pd.concat([results, pd.DataFrame({model: predictions})], axis=1)

        return results