import numpy as np
from collections import defaultdict

class KMeansClustering:

    def __init__(self):
        self.centroids = None
        self.cluster = None

    def train(self, training_data: np.ndarray, k: int = 3, iters: int = 5):
        centroid_idx = np.random.choice(training_data.shape[0], k, replace=False)
        centroids = training_data[centroid_idx]
        for _ in range(iters):
            cluster = defaultdict(list)
            for point in training_data:
                assigned_centroid = self.get_assigned_centroid(point, centroids)
                cluster[tuple(assigned_centroid)].append(point)
            centroids = [np.sum(cluster[key], axis=0)/len(cluster[key]) for key in cluster.keys()]
        self.centroids = centroids
        self.cluster = cluster

    def classify(self, eval_data: np.ndarray):
        final_centroids = []
        for point in eval_data:
            final_centroids.append(self.get_assigned_centroid(point, self.centroids))
        return final_centroids

    def get_assigned_centroid(self, point: np.ndarray, centroids: np.ndarray):
        assigned_centroid = None
        min_distance = float('inf')
        for centroid in centroids:
            if np.linalg.norm(centroid-point) < min_distance:
                assigned_centroid = centroid
                min_distance = np.linalg.norm(centroid-point)
        return assigned_centroid