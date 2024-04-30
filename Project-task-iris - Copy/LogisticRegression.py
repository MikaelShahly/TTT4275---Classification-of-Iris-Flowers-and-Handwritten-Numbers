import numpy as np 
import pandas as pd 

class LinearClassifier:
    
    def __init__(self, learning_rate=0.008, iterations=10000):
        """
        sets the hyperparamaters for the model
        Args:
            learning_rate (float): specifies the learning rate used in
              gradient descent to fit the model.
            iterations (int): specifies the amount of iterations used in
              gradient descent to fit the model.

        """
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def fit_sigmoid(self, X, y):
        """
        Estimates weights and biases for the classifier using a sigmoid activation function
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (matrix<m>): a vector of ints containing 
                m label values
        """
        #initialize paramaters
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
 
        self.W = np.zeros((n_features, n_classes))
        self.B = np.zeros(n_classes)

        #convert lables to a matrix<m, n_classes> with 1 column for each class
        y_dummy = pd.get_dummies(y, prefix='class')
        y_dummy = y_dummy.values

        alpha_vals = []

        #perform gradient descent
        for i in range(self.iterations):
            Z_test = np.dot(X, self.W)
            B_test = self.B
            Z = np.dot(X, self.W) + self.B
            MSE_sigmoid = (1/2) * np.sum((sigmoid(Z) - y_dummy) ** 2)

            dW = np.dot(X.T, ((sigmoid(Z) * (1 - sigmoid(Z))) * (sigmoid(Z) - y_dummy)))
            dB = np.sum((sigmoid(Z) * (1 - sigmoid(Z))) * (sigmoid(Z) - y_dummy), axis=0)   

            self.W = self.W - self.learning_rate * dW
            self.B = self.B - self.learning_rate * dB
           
            if (i%100 == 0):
                print(f"cost after {i} iterations of gradient descent is: {MSE_sigmoid}")
                alpha_vals.append(MSE_sigmoid)
        return alpha_vals

    def fit_linear(self, X, y):
        """
        Estimates weights and biases for a linear classifier without a sigmoid activation function
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (matrix<m>): a vector of ints containing 
                m label values
        """

        #initialize paramaters
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.W = np.random.randn(n_features, n_classes)
        self.B = np.random.randn(n_classes)

        #convert lables to a matrix<m, n_classes> with 1 column for each class
        y_dummy = pd.get_dummies(y, prefix='class')
        y_dummy = y_dummy.values


        #perform gradient descent
        for i in range(self.iterations):
            Z = np.dot(X, self.W) + self.B
            MSE = (1/(n_samples)) * np.sum((Z - y_dummy) ** 2 )

            dW = (2 / (n_samples)) * np.dot(X.T, (Z - y_dummy))
            dB = (2 / (n_samples)) * np.sum((Z - y_dummy), axis=0)   

            self.W = self.W - self.learning_rate * dW
            self.B = self.B - self.learning_rate * dB

            if (i%100 == 0):
                print(f"cost after {i} iterations of gradient descent is: {MSE}")

    
    def predict_class_probability(self, X):
        """
        Generates probability predictions if using the model trained with a sigmoid activation function
        
        Note: should be called after .fit_sigmoid()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        z = np.dot(X, self.W) + self.B
        return sigmoid(z)
    
    def predict_class_sigmoid(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit_sigmoid()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with class predictions
        """
        z = np.dot(X, self.W) + self.B
        # Predict the class based on the maximum value along each row
        predictions = np.argmax(sigmoid(z), axis=1) + 1  # Adding 1 to make the prediction 1, 2, 3 npot 0,1,2
        return predictions
    
    def predict_class_linear(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit_linear()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with class predictions
        """

        Z = np.dot(X, self.W) + self.B

        # Predict the class based on the maximum value along each row
        predictions = np.argmax(Z, axis=1) + 1  # Adding 1 to make the prediction 1, 2, 3 not 0,1,2
        return predictions



def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """

    return 1. / (1. + np.exp(-x))


