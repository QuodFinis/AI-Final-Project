# John Royal
# CSc 44800: Artificial Intelligence - Final Project
# This file contains functions for analyzing the results of the K-Means algorithm.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_predicted_labels(clusters: np.ndarray, labels: np.ndarray):
    """
    Finds the most common label in each cluster, and returns a list of predicted labels for each data point.
    Also returns a list of the most common label in each cluster, and prints the accuracy for each cluster.
    """
    cluster_labels = []

    for i in range(10):
        # Find the index of points in cluster i.
        indices = np.where(clusters == i)[0]
        labels_in_cluster = labels[indices]

        # Find the most common label in the cluster.
        if len(labels_in_cluster) > 0:
            most_common = np.bincount(labels_in_cluster).argmax()
            cluster_labels.append(most_common)
        else:
            cluster_labels.append(None)

    # Calculate accuracy for each cluster.
    for i in range(10):
        indices = np.where(clusters == i)[0]
        labels_in_cluster = labels[indices]
        correct_labels = labels_in_cluster == cluster_labels[i]
        accuracy = np.sum(correct_labels) / len(labels_in_cluster)
        print(
            f"  Cluster {i} (Most Common Digit: {cluster_labels[i]}) - Accuracy: {accuracy:.2f}"
        )

    predicted_labels = [cluster_labels[cluster] for cluster in clusters]

    return predicted_labels, cluster_labels


def compute_accuracy(test_labels: np.ndarray, predicted_labels: list[int]):
    """
    Computes the accuracy of the predicted labels.
    """
    correct_labels = predicted_labels == test_labels
    accuracy = np.sum(correct_labels) / len(test_labels)
    return accuracy


def compute_confusion_matrix(test_labels: np.ndarray, predicted_labels: list[int]):
    """
    Computes the confusion matrix for the predicted labels.
    """
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for i in range(len(test_labels)):
        confusion_matrix[predicted_labels[i], test_labels[i]] += 1
    return confusion_matrix


def plot_confusion_matrix(
    title: str, cm: np.ndarray, test_labels: np.ndarray, predicted_labels: list[int]
):
    """
    Plots the confusion matrix for the predicted labels.
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=0.5, square=True, cmap="Blues")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    complete_title = (
        f"{title} - Accuracy: {compute_accuracy(test_labels, predicted_labels):.2f}"
    )
    plt.title(complete_title, size=15)
    plt.show()
