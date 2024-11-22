import tensorflow as tf
from tensorflow.keras import Sequential  # Used to define a sequential model
from tensorflow.keras.layers import Flatten, Dense  # Layers used in our neural network
from tensorflow.keras.datasets import fashion_mnist  # Dataset used for training and testing
import numpy as np  # For numerical operations
import unittest  # Framework for unit testing

class FashionImageClassifier:
    """
    A class that encapsulates the functionality of a neural network-based image classifier 
    for identifying fashion items. The model is trained on the Fashion MNIST dataset, which
    includes images of clothing items such as shirts, shoes, and dresses.
    """
    
    def __init__(self):
        """
        Constructor method to initialize the classifier. It loads the Fashion MNIST dataset
        and sets up variables to hold the training/testing data and the model itself.
        """
        # Load Fashion MNIST dataset. This dataset is divided into training and testing sets.
        # Each image is 28x28 pixels in grayscale.
        (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
        
        # Placeholder for the neural network model.
        self.model = None

    def preprocess_data(self):
        """
        Preprocesses the dataset by normalizing the pixel values of the images to the range [0, 1].
        This improves model training by ensuring consistent input data scaling.
        """
        # Normalize the pixel values of the images (originally in range 0-255) to range [0, 1].
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    def build_model(self):
        """
        Builds a simple feedforward neural network using TensorFlow's Keras API. 
        The model consists of an input layer (Flatten), one hidden layer (Dense), and an output layer (Dense).
        """
        # The Sequential model allows us to stack layers in a linear fashion.
        self.model = Sequential([
            # The Flatten layer reshapes the 28x28 image into a 1D array of 784 features.
            Flatten(input_shape=(28, 28)),  
            # A Dense layer with 128 neurons and ReLU activation function.
            Dense(128, activation='relu'),
            # Output layer with 10 neurons (one for each fashion category) and softmax activation for probabilities.
            Dense(10, activation='softmax')
        ])
        
        # Compile the model by specifying the optimizer, loss function, and evaluation metrics.
        self.model.compile(
            optimizer='adam',  # Adaptive Moment Estimation (Adam) optimizer, commonly used for deep learning.
            loss='sparse_categorical_crossentropy',  # Suitable for multi-class classification problems.
            metrics=['accuracy']  # Metric to monitor the accuracy during training.
        )

    def train_model(self, epochs=5):
        """
        Trains the model using the training dataset. 
        The number of epochs determines how many complete passes are made over the training data.
        """
        if self.model is None:
            raise Exception("Model not built yet. Call build_model first.")
        # Fit the model to the training data.
        self.model.fit(self.x_train, self.y_train, epochs=epochs)

    def evaluate_model(self):
        """
        Evaluates the model's performance using the test dataset and returns the loss and accuracy metrics.
        """
        if self.model is None:
            raise Exception("Model not built yet. Call build_model first.")
        # Evaluate the model on unseen test data and return the results.
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, accuracy

    def predict(self, image):
        """
        Predicts the class label of a single image using the trained model.
        
        Parameters:
        - image: A 2D numpy array representing a grayscale image of size 28x28.
        
        Returns:
        - The predicted class label as an integer (0-9).
        """
        if self.model is None:
            raise Exception("Model not built yet. Call build_model first.")
        # Preprocess the input image: expand dimensions and normalize pixel values.
        image = np.expand_dims(image, axis=0) / 255.0  
        # Get model predictions (probabilities for each class).
        predictions = self.model.predict(image)
        # Return the class with the highest probability.
        return np.argmax(predictions)

# Unit tests to validate the functionality of the FashionImageClassifier
class TestFashionImageClassifier(unittest.TestCase):
    """
    Unit test class to test various functionalities of the FashionImageClassifier.
    Uses the unittest module to define and run test cases.
    """
    
    def setUp(self):
        """
        Set up a classifier instance and prepare it for testing.
        """
        self.classifier = FashionImageClassifier()
        self.classifier.preprocess_data()
        self.classifier.build_model()

    def test_model_training(self):
        """
        Tests whether the model can be trained successfully and achieves a basic accuracy threshold.
        """
        self.classifier.train_model(epochs=1)  # Train with 1 epoch for quick testing.
        loss, accuracy = self.classifier.evaluate_model()
        # Check if accuracy exceeds a basic threshold (50% for this example).
        self.assertGreater(accuracy, 0.5, "Accuracy should be greater than 50% for a simple test.")

    def test_prediction(self):
        """
        Tests the model's ability to make predictions on sample data.
        """
        self.classifier.train_model(epochs=1)  # Train with 1 epoch for quick testing.
        image = self.classifier.x_test[0]  # Use the first test image.
        prediction = self.classifier.predict(image)
        # Validate that the prediction is an integer and a valid class index.
        self.assertIsInstance(prediction, int, "Prediction should be an integer.")
        self.assertIn(prediction, range(10), "Prediction should be a valid class index.")

