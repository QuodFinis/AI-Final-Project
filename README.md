# CSc 44800: Artificial Intelligence - Final Project for Team MENJ

This is Team MENJ (Mahmud Hasan, John Royal, Nur Mazumder, Erlin Paredes)’s final project for CSc 44800.

We created a system that recognizes text using the MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits converted into black and white, 28x28 pixel format.

## Setup

To start, use the following command to install the required dependencies using Anaconda and our `requirements.txt`:

```
conda install --file requirements.txt
```

Then, `cd` into each group member’s respective directory. If any specific instructions are required to run their portion of the project, they will be included in their `README.md`. For example:

```
cd Erlin-Paredes-KNN
python main.py
```

## Libraries

Per our `requirements.txt`, we used the following libraries:

- `numpy`
- `scikit-learn`
- `scikit-image`
- `matplotlib`
- `seaborn`
- `opencv`
- `pandas`
- `pytorch`
- `tensorflow`
- `torchvision`

## Models

Each group member implemented text recognition using a different type of model.

| Model | Implemented By | Directory |
| - | - | - |
| K-Means | John Royal | `K-Means_John-Royal` |
| KNN | Erlin Paredes | `KNN_Erlin-Paredes` |
| Logistic Regression, Naive Bayes, CNN | Nur Mazumder |  `Logistic-Regression_Nur-Mazumder` |
| Neural Network | Mahmud Hasan | `Neural-Network_Mahmud-Hasan` |

### K-Means (John Royal)

[Description here]

### KNN (Erlin Paredes)

[Description here]

### Logistic Regression, Naive Bayes, CNN (Nur Mazumder)

Implemented on Jupyter Notebook, you can run the notebooks as with the usual libraries. For image recognition using our own handwritten digits, make sure the image.pngs are in the same directories. 

### Neural Network (Mahmud Hasan)

Neural network is built from scratch using mostly numpy. To run and test simply run the neuralnet.py. Neural network built using PyTorch is created similar to how the one from scratch is built. An autoencoder network is also built to attempt to recreate images into mnist database, which may help train a neural network that can run on less data. Additionally, there is a generator which attempts to use the neural network model in reverse and create new handwritten digits of a similar type and style as the mnist database, however it is still very inaccurate. Model is in model.py and will show some random testcases and its output.

