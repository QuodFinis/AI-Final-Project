# John Royal
# CSc 44800: Artificial Intelligence - Final Project
# This file contains the K-Means algorithm, implemented from scratch.

import numpy as np


class KMeans:
    def __init__(self, n_clusters: int, n_init: int, max_iter: int, random_state: int):
        self.best_inertia = np.inf
        self.best_centroids = None
        self.best_labels = None

        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray[np.float32]):
        """
        Runs the K-Means algorithm on the given dataset, and stores the best centroids and labels.
        """

        # Set random seed if specified.
        if self.random_state:
            np.random.seed(self.random_state)

        # Run the algorithm n_init times, and store the best result.
        for _ in range(self.n_init):
            # A centroid is a point in the dataset.
            # The initial centroids are chosen randomly from the dataset.
            centroids = self._initialize_centroids(X)

            # Run the algorithm max_iter times, or until the centroids stop moving.
            for _ in range(self.max_iter):
                # Assign each point to the closest centroid.
                labels = self._closest_centroid(X, centroids)
                # Move the centroids to the center of their assigned points.
                new_centroids = self._move_centroids(X, labels, centroids)
                # If the centroids didn't move, we're done.
                if np.all(centroids == new_centroids):
                    break
                # Otherwise, save the new centroids and continue.
                centroids = new_centroids

            # Inertia is the sum of the squared distances between each point and its centroid.
            # The lower the inertia, the better the clustering.
            # This computes the inertia for the current run.
            inertia = np.sum((X - centroids[labels]) ** 2)
            # If the inertia is lower than the best inertia so far,
            # save the centroids and labels from this run as the best.
            if inertia < self.best_inertia:
                self.best_inertia = inertia
                self.best_centroids = centroids
                self.best_labels = labels

    def predict(self, X: np.ndarray[np.float32]):
        """
        Predicts the cluster for each data point in X.
        """
        return self._closest_centroid(X, self.best_centroids)

    def _initialize_centroids(self, points: np.ndarray[np.float32]):
        """
        Randomly initializes centroids.
        """
        centroids = points.copy()
        np.random.shuffle(centroids)
        return centroids[: self.n_clusters]

    def _closest_centroid(self, points: np.ndarray[np.float32], centroids: np.ndarray):
        """
        Returns an array containing the index to the nearest centroid for each point.
        """
        distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _move_centroids(
        self, points: np.ndarray[np.float32], closest: np.ndarray, centroids: np.ndarray
    ):
        """
        Returns the new centroids assigned from the points closest to them.
        """
        return np.array(
            [points[closest == k].mean(axis=0) for k in range(centroids.shape[0])]
        )
