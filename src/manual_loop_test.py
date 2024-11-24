import matplotlib.pyplot as plt
from classifier import FashionImageClassifier

def visualize_predictions():
    classifier = FashionImageClassifier()
    classifier.preprocess_data()
    classifier.build_model()
    classifier.train_model(epochs=10)

    for i in range(100):
        image = classifier.x_test[i]
        prediction = classifier.predict(image)
        plt.imshow(image, cmap='gray')
        plt.title(str(classifier.class_names()[prediction]))
        plt.show()

if __name__ == "__main__":
    visualize_predictions()