import unittest
from src.classifier import FashionImageClassifier

class TestFashionImageClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = FashionImageClassifier()
        self.classifier.preprocess_data()
        self.classifier.build_model()

    def test_model_training(self):
        self.classifier.train_model(epochs=1)
        loss, accuracy = self.classifier.evaluate_model()
        self.assertGreater(accuracy, 0.5, "Accuracy should be greater than 50%.")

    def test_prediction(self):
        self.classifier.train_model(epochs=1)
        image = self.classifier.x_test[0]
        prediction = self.classifier.predict(image)
        self.assertIsInstance(prediction, int)
        self.assertIn(prediction, range(10))

if __name__ == "__main__":
    unittest.main()
