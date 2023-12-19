Team MENJ: Mahmud Hasan, John Royal, Nur Mazumder, Erlin Paredes
    
   
This project aims to create a system that can recognize text.


I will be using the MNIST dataset, which contains 60,000 training images
and 10,000 testing images of handwritten digits, converted into black and
white, 28x28 pixel format.
  
  
Libraries used
Pandas    
NumPy    
Matplotlib   
Scikit-learn  
  
  
Models Used    
K-means    
K-nearest-neighbors    
Logistic regression    
Neural Network   

Modeling: K-means |   

Modeling: K-nearest-neighbors |   

Modeling: Logistic Regression |   

Modeling: Neural Netwrok | Mahmud Hasan   
Neural network is built from scratch using mostly numpy. To run and test simply run the neuralnet.py. Neural network built using PyTorch is created similar to how the one from scratch is built. An autoencoder network is also built to attempt to recreate images into mnist database, which may help train a neural network that can run on less data. Additionally, there is a generator which attempts to use the neural network model in reverse and create new handwritten digits of a similar type and style as the mnist database, however it is still very inaccurate.  
