"""
Module docstring
"""

from typing import List
from keras import Sequential
from keras.layers import Dense
import numpy as np

class NeuralNetwork:
    """_summary_
    """

    def __init__(self):
        """
        Initialize neural network with:
        - 13 units in input layer
        - 10 units in hidden dense layer with relu activation function
        - 1 unit in output dense layer with sigmoid activation function
        """        
        self.model = Sequential([
            Dense(10, input_shape=(13,), activation="relu"),
            Dense(1, activation="sigmoid")
        ])

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set given weights to neural network

        Args:
            weights (np.ndarray): Numpy array of shape (151,) contains weights of neural network to set
                                  0 - 129     weights between input and hidden layer
                                  130 - 139   biases of hidden layer
                                  140 - 149   weights between hidden and output layer
                                  150         bias of output layer

        Raises:
            ValueError: raise if shape of weights if other than (151,)
        """        
    
        if weights.shape != (151,):
            raise ValueError('Shape of weights must be (151,)')

        weight_1 = weights[0:130].reshape((13,10))
        bias_1 = weights[130:140].reshape((10,))
        weight_2 = weights[140:150].reshape((10,1))
        bias_2 = weights[150].reshape((1,))

        parameters = [weight_1, bias_1, weight_2, bias_2]
        self.model.set_weights(parameters)
    
    def get_weights(self) -> List[np.ndarray]:
        """
        Return weights of neural network

        Returns:
            List[np.ndarray]: List contains 4 np.ndarray with weights and biases
        """        
        return self.model.get_weights()

    def get_accuracy(self, x: np.ndarray, y: np.ndarray, batch_size=10) -> float:
        """
        Return accuracy of model based on input arrays

        Args:
            x (np.ndarray): feature data
            y (np.ndarray): labels
            batch_size (int, optional): batch size. Defaults to 10.

        Returns:
            float: accuracy value
        """         
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        acc = self.model.evaluate(x, y, batch_size=batch_size, verbose=0)[1]
        return acc

    def get_loss(self, x: np.ndarray, y: np.ndarray, batch_size=10) -> float:
        """
        Return loss of model based on input arrays

        Args:
            x (np.ndarray): feature data
            y (np.ndarray): labels
            batch_size (int, optional): batch size. Defaults to 10.

        Returns:
            float: loss value
        """        
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        loss = self.model.evaluate(x, y, batch_size=batch_size, verbose=0)[0]
        return loss