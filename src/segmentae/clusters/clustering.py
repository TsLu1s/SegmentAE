from sklearn.cluster import (KMeans,
                             MiniBatchKMeans,
                             AgglomerativeClustering)
from sklearn.mixture import GaussianMixture

class KMeans_Clustering:
    def __init__(self, 
                 n_clusters=3, 
                 random_state=0, 
                 max_iter=300):
        """
        KMeans_Clustering is a class for performing K-Means clustering.

        Parameters:
        - n_clusters (int): The number of clusters to form.
        - random_state (int): Random seed for initializing centroids.
        - max_iter (int): Maximum number of iterations to run.

        Attributes:
        - n_clusters (int): The number of clusters to form.
        - random_state (int): Random seed for initializing centroids.
        - max_iter (int): Maximum number of iterations to run.
        - model (KMeans): K-Means clustering model.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = KMeans(
            n_clusters = self.n_clusters,
            random_state = self.random_state,
            max_iter = self.max_iter,
        )

    def fit(self, X):
        """
        Fit the KMeans model to the data.

        Parameters:
        - X (array-like): The input data.

        Returns:
        - None
        """
        self.__init__(self.n_clusters, self.random_state, self.max_iter)
        self.model.fit(X)

    def predict(self, X):
        """
        Predict cluster labels for the given data.

        Parameters:
        - X (array-like): The input data.

        Returns:
        - labels (array): Predicted cluster labels.
        """
        return self.model.predict(X)

class MiniBatchKMeans_Clustering:
    def __init__(self, 
                 n_clusters=3, 
                 random_state=0, 
                 max_iter=150):
        """
        MiniBatchKMeans_Clustering is a class for performing Mini Batch K-Means clustering.

        Parameters:
        - n_clusters (int): The number of clusters to form.
        - random_state (int): Random seed for initializing centroids.
        - max_iter (int): Maximum number of iterations to run.

        Attributes:
        - n_clusters (int): The number of clusters to form.
        - random_state (int): Random seed for initializing centroids.
        - max_iter (int): Maximum number of iterations to run.
        - model (MiniBatchKMeans): Mini Batch K-Means clustering model.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = MiniBatchKMeans(
            n_clusters = self.n_clusters,
            random_state = self.random_state,
            max_iter = self.max_iter,
        )

    def fit(self, X):
        """
        Fit the MiniBatchKMeans model to the data.

        Parameters:
        - X (array-like): The input data.

        Returns:
        - None
        """
        self.__init__(self.n_clusters, self.random_state, self.max_iter)
        self.model.fit(X)

    def predict(self, X):
        """
        Predict cluster labels for the given data.

        Parameters:
        - X (array-like): The input data.

        Returns:
        - labels (array): Predicted cluster labels.
        """
        return self.model.predict(X)   
    
class GaussianMixture_Clustering:
    def __init__(self, 
                 init_params='k-means++',
                 n_components=3,
                 max_iter=150,
                 covariance_type='tied'):
        """
        GaussianMixture_Clustering is a class for performing Gaussian Mixture Model clustering.

        Parameters:
        - init_params (str): Method for initialization of the means and the precisions. {'kmeans', 'k-means++', 'random', 'random_from_data'}
        - n_components (int): Number of mixture components.
        - max_iter (int): Maximum number of EM iterations.
        - covariance_type (str): Type of covariance parameters to use. {'full', 'tied', 'diag', 'spherical'}

        Attributes:
        - init_params (str): Method for initialization of the means and the precisions. {'kmeans', 'k-means++', 'random', 'random_from_data'}
        - n_components (int): Number of mixture components.
        - max_iter (int): Maximum number of EM iterations.
        - covariance_type (str): Type of covariance parameters to use. {'full', 'tied', 'diag', 'spherical'}
        - model (GaussianMixture): Gaussian Mixture Model clustering model.
        """
        self.init_params = init_params
        self.n_components = n_components
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.model = GaussianMixture(
            n_components = self.n_components,
            max_iter = self.max_iter,
            covariance_type = self.covariance_type,
        )

    def fit(self, X):
        """
        Fit the GaussianMixture model to the data.

        Parameters:
        - X (array-like): The input data.

        Returns:
        - None
        """
        self.__init__(self.init_params, 
                      self.n_components,
                      self.max_iter,
                      self.covariance_type)
        self.model.fit(X)

    def predict(self, X):
        """
        Predict cluster labels for the given data.

        Parameters:
        - X (array-like): The input data.

        Returns:
        - labels (array): Predicted cluster labels.
        """
        return self.model.predict(X) 

class Agglomerative_Clustering:
    def __init__(self, 
                 n_clusters=3, 
                 linkage='ward', 
                 distance_threshold=None):
        """
        Agglomerative_Clustering is a class for performing Agglomerative Clustering.

        Parameters:
        - n_clusters (int): The number of clusters to find.
        - linkage (str): The linkage criterion to use.
        - distance_threshold (float): The linkage distance threshold above which, clusters will not be merged.

        Attributes:
        - n_clusters (int): The number of clusters to find.
        - linkage (str): The linkage criterion to use.
        - distance_threshold (float): The linkage distance threshold above which, clusters will not be merged.
        - model (AgglomerativeClustering): Agglomerative Clustering model.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold

    def fit(self, X):
        """
        Fit the AgglomerativeClustering model to the data.

        Parameters:
        - X (array-like): The input data.

        Returns:
        - None
        """
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             linkage=self.linkage,
                                             distance_threshold=self.distance_threshold)
        self.model.fit(X)

    def predict(self, X):
        """
        Predict cluster labels for the given data.

        Parameters:
        - X (array-like): The input data.

        Returns:
        - labels (array): Predicted cluster labels.
        """
        return self.model.fit_predict(X) #.labels_  