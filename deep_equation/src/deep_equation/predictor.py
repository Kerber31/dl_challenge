"""
Predictor interfaces for the Deep Learning challenge.
"""

from tensorflow import keras
from tensorflow.image import resize
from typing import List
import numpy as np


class BaseNet:
    """
    Base class that must be used as base interface to implement 
    the predictor using the model trained by the student.
    """

    def load_model(self, model_path):
        """
        Implement a method to load models given a model path.
        """
        pass

    def predict(
        self, 
        images_a: List, 
        images_b: List, 
        operators: List[str], 
        device: str = 'cpu'
    ) -> List[float]:
        """
        Make a batch prediction considering a mathematical operator 
        using digits from image_a and image_b.
        Instances from iamges_a, images_b, and operators are aligned:
            - images_a[0], images_b[0], operators[0] -> regards the 0-th input instance
        Args: 
            * images_a (List[PIL.Image]): List of RGB PIL Image of any size
            * images_b (List[PIL.Image]): List of RGB PIL Image of any size
            * operators (List[str]): List of mathematical operators from ['+', '-', '*', '/']
                - invalid options must return `None`
            * device: 'cpu' or 'cuda'
        Return: 
            * predicted_number (List[float]): the list of numbers representing the result of the equation from the inputs: 
                [{digit from image_a} {operator} {digit from image_b}]
        """
    # do your magic

    pass 


class RandomModel(BaseNet):
    """This is a dummy random classifier, it is not using the inputs
        it is just an example of the expected inputs and outputs
    """

    def load_model(self, model_path):
        """
        Method responsible for loading the model.
        If you need to download the model, 
        you can download and load it inside this method.
        """
        np.random.seed(42)

    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ) -> List[float]:

        predictions = []
        for image_a, image_b, operator in zip(images_a, images_b, operators):            
            random_prediction = np.random.uniform(-10, 100, size=1)[0]
            predictions.append(random_prediction)
        
        return predictions


class StudentModel(BaseNet):

    def __init__(self):
        self.model = None

    def load_model(self, model_path: str = 'model/model.h5'):
        self.model = keras.models.load_model(model_path)
    
    def _one_hot_operator(self, operator):
        if operator == '+':
            return np.array([1, 0, 0, 0])
        elif operator == '-':
            return np.array([0, 1, 0, 0])
        elif operator == '*':
            return np.array([0, 0, 1, 0])
        else:
            return np.array([0, 0, 0, 1])

    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ):
        if self.model == None:
            self.load_model()
    
        predictions = []
        processed_a = []
        processed_b = []
        processed_op = []

        for i in range(len(images_a)):
            image_a = (np.array(images_a[i])[:, :, :3]/255 - 0.5)*2
            image_b = (np.array(images_b[i])[:, :, :3]/255 - 0.5)*2
            operator = np.array(operators[i])

            processed_a.append(resize(image_a, [32, 32], method='nearest'))
            processed_b.append(resize(image_b, [32, 32], method='nearest'))
            processed_op.append(self._one_hot_operator(operator))

        processed_a = np.array(processed_a)
        processed_b = np.array(processed_a)
        processed_op = np.array(processed_op)
        
        predictions = self.model([processed_a, processed_b, processed_op]).numpy().tolist()
        predictions = [element[0] for element in predictions]

        return predictions
