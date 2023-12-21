# John Royal
# CSc 44800: Artificial Intelligence - Final Project
# This file contains functions for loading the MNIST dataset.

import numpy as np

from array import array
from pathlib import Path
from os.path import join
from struct import unpack


class MNISTRawData:
    DATASET_PATH = join(Path(__file__).parent, "dataset")

    @staticmethod
    def load_training_data():
        train_images_path = join(
            MNISTRawData.DATASET_PATH, "train-images-idx3-ubyte/train-images-idx3-ubyte"
        )
        train_labels_path = join(
            MNISTRawData.DATASET_PATH, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
        )
        return MNISTRawData(train_images_path, train_labels_path)

    @staticmethod
    def load_testing_data():
        test_images_path = join(
            MNISTRawData.DATASET_PATH, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
        )
        test_labels_path = join(
            MNISTRawData.DATASET_PATH, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
        )
        return MNISTRawData(test_images_path, test_labels_path)

    def __init__(self, images_path: str, labels_path: str):
        self.images = self._load_images(images_path)
        self.labels = self._load_labels(labels_path)

    @staticmethod
    def _load_images(images_file_path: str):
        """
        Loads images from the given file path.
        """

        with open(images_file_path, "rb") as file:
            # Unpack the first 16 bytes of the file, which contain the following metadata:
            # - The magic number, which should be 2051 to indicate that this is an image file.
            # - The number of images in the dataset.
            # - The number of rows in each image.
            # - The number of columns in each image.
            magic_num, size, rows, cols = unpack(">IIII", file.read(16))
            if magic_num != 2051:
                raise ValueError(
                    f"Unexpected magic number: expected 2051 but found {magic_num}"
                )
            
            # Read the rest of the file, which contains the actual image data.
            image_data = array("B", file.read())

        return image_data, size, rows, cols

    @staticmethod
    def _load_labels(labels_file_path: str):
        """
        Loads labels from the given file path.
        """

        with open(labels_file_path, "rb") as file:
            # Unpack the first 8 bytes of the file, which contain the following metadata:
            # - The magic number, which should be 2049 to indicate that this is a label file.
            # - The number of labels in the dataset.
            magic_num, size = unpack(">II", file.read(8))
            if magic_num != 2049:
                raise ValueError(
                    f"Unexpected magic number: expected 2049 but found {magic_num}"
                )
            
            # Read the rest of the file, which contains the actual label data.
            labels = np.array(array("B", file.read()))

        return labels
