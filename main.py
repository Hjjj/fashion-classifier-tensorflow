from src.classifier import FashionImageClassifier

if __name__ == "__main__":
    # Instantiate the classifier and demonstrate training and evaluation.
    classifier = FashionImageClassifier()
    classifier.preprocess_data()
    classifier.build_model()
    classifier.train_model(epochs=5)  # Train the model for 5 epochs.
    loss, accuracy = classifier.evaluate_model()
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Run the unit tests to validate functionality.
    #unittest.main(argv=[''], verbosity=2, exit=False)
