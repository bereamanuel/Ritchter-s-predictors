import numpy as np
from sklearn.cluster import KMeans

class KMeansFeaturizer:
    """Transforms numeric data into k-means cluster memberships.
    
    This transformer runs k-means on the input data and converts each data point
    into the ID of the closest cluster. If a target variable is present, it is
    scaled and included as input to k-means in order to derive clusters that
    obey the classification boundary as well as group similar points together.
    """
    
    def __init__(self, k=100, target_scale=5.0, random_state=None):
        self.k = k
        self.target_scale = target_scale
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """
        Runs k-means on the input data and finds centroids.
        """
        if y is None:
        # No target variable, just do plain k-means
            km_model = KMeans(n_clusters=self.k,
                        n_init=20,
                        random_state=self.random_state)
            km_model.fit(X)
            
            self.km_model_ = km_model
            self.cluster_centers_ = km_model.cluster_centers_
            return self
        
        # There is target information. Apply appropriate scaling and include
        # it in the input data to k-means.
        data_with_target = np.hstack((X, y[:,np.newaxis]*self.target_scale))
        
        # Build a pre-training k-means model on data and target
        km_model_pretrain = KMeans(n_clusters=self.k,
                                    n_init=20,
                                    random_state=self.random_state)

        km_model_pretrain.fit(data_with_target)
        
        # Run k-means a second time to get the clusters in the original space
        # without target info. Initialize using centroids found in pre-training.
        # Go through a single iteration of cluster assignment and centroid
        # recomputation.

        n = X.shape[1]

        km_model = KMeans(n_clusters=self.k,
                                init=km_model_pretrain.cluster_centers_[:,:n],
                                n_init=1,
                                max_iter=1)
        km_model.fit(X)
        

        self.km_model = km_model
        self.cluster_centers_ = km_model.cluster_centers_
        self.score = km_model.score(X)
        return self
    
    def transform(self, X, y=None):
        """Outputs the closest cluster ID for each input data point.
        """
        clusters = self.km_model.predict(X)
        return clusters[:,np.newaxis]
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)