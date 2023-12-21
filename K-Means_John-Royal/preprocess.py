# John Royal
# CSc 44800: Artificial Intelligence - Final Project
# This file contains functions for preprocessing the MNIST dataset.

import numpy as np
from load import MNISTRawData
from skimage.transform import rotate


class MNISTPreprocessedData:
    """
    A class that represents the MNIST dataset after preprocessing (i.e. flattening and normalizing the images).
    This also includes methods for further preprocessing, such as augmentation.
    """

    @staticmethod
    def from_raw_data(raw_data: MNISTRawData):
        """
        Returns a new MNISTPreprocessedData object from the given MNISTRawData object.
        """
        raw_images, size, rows, cols = raw_data.images

        # Convert the images from a 28 x 28 matrix to a 784 x 1 vector,
        # and normalize the values to be between 0 and 1.
        images = np.zeros((size, rows * cols), dtype=np.float32)
        for i in range(size):
            img = np.array(raw_images[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i, :] = img.flatten() / 255.0

        return MNISTPreprocessedData(
            images,
            raw_data.labels,
        )

    def __init__(self, images: np.ndarray[np.float32], labels: np.ndarray[np.uint8]):
        self.images = images
        self.labels = labels

    def augmented(self):
        """
        Returns a new MNISTPreprocessedData object with augmented images.
        The augmentation is done by rotating each image slightly and adding it to the dataset, along with the original.
        """
        augmented_images = []
        augmented_labels = []

        for label, image in zip(self.labels, self.images):
            # Append original image and label.
            augmented_images.append(image)
            augmented_labels.append(label)

            # Reshape the flattened image back to 2D.
            img_2d = image.reshape(28, 28)

            # Rotate the image slightly.
            rotated = rotate(img_2d, angle=np.random.uniform(-10, 10), mode="wrap")

            # Flatten back and append.
            augmented_images.append(rotated.flatten())
            augmented_labels.append(label)

        return MNISTPreprocessedData(
            np.array(augmented_images),
            np.array(augmented_labels),
        )
