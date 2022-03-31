"""
TODO Module docstring
"""

#pylint: disable=invalid-name

from typing import List
from keras import Sequential
from keras.layers import Dense
from keras.utils.layer_utils import count_params
import numpy as np
from decorators import measure_time

class NeuralNetwork:
    """ TODO class docstring
    """

    def __init__(self, num_of_features):

        self.model = Sequential([
            Dense(10, input_shape=(num_of_features,), activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def get_total_parameters_count(self):
        total_parameters_count = count_params(self.model.trainable_weights)
        return total_parameters_count

    def set_weights(self, weights: np.ndarray) -> None:
        """ TODO

        Args:
            weights (np.ndarray): _description_

        Raises:
            ValueError: _description_
        """
        total_parameters_count = count_params(self.model.trainable_weights)
        shapes_of_weights = [elem.shape for elem in self.get_weights()]

        if weights.shape != (total_parameters_count,):
            raise ValueError(f'Shape of weights must be ({total_parameters_count},)')

        parameters = []
        current_index = 0
        for inx, elem in enumerate(shapes_of_weights):
            parameters_size = elem[0] if len(elem) == 1 else elem[0] * elem[1]
            weight = weights[current_index:current_index+parameters_size].reshape(shapes_of_weights[inx])
            parameters.append(weight)
            current_index += parameters_size

        self.model.set_weights(parameters)

    def get_weights(self) -> List[np.ndarray]:
        """ TODO

        Returns:
            List[np.ndarray]: _description_
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

        loss = self.model.evaluate(x, y, batch_size=batch_size, verbose=0)[0]
        return loss