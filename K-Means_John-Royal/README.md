# Optical Character Recognition with K-Means Clustering - John Royal

This directory contains my implementation of optical character recognition (OCR) using the K-Means clustering algorithm.

This includes the following files (in order of execution):

1. `load.py`: Loads the MNIST dataset from the `dataset` directory. This is used for both training and testing data.
2. `preprocess.py`: Preprocesses the MNIST dataset. This includes flattening and normalizing the images, as well as augmenting the training dataset with rotated images.
3. `kmeans.py`: Contains my implementation of the K-Means algorithm from scratch.
4. `analysis.py`: Interprets the output from the K-Means algorithm (i.e. determining what the predicted labels actually are), computes the accuracy, and creates and plots a confusion matrix.

These are all implemented using classes and functions, which are called from `main.ipynb`. This main file runs both my implementation of K-Means from scratch and the implementation from Scikit-Learn, then runs the analysis and plots the confusion matrices for both.

To run this:
1. Make sure you have the dependencies from the `requirements.txt` in the main directory. 
2. Consider adjusting the configuration constants at the top of `main.ipynb`, especially the `MAX_ITER` value. With the current value, 100, the from-scratch implementation took over 15 minutes to run.
3. Run `main.ipynb`.
