import numpy as np
import matplotlib.ticker as mticker
from sklearn.metrics import ConfusionMatrixDisplay

# With Regularization
class LogisticRegression:
    def __init__(self, alpha=0.01, iterations=1000, lambdaparam=0.01):
        self.alpha = alpha
        self.iterations = iterations
        self.lambdaparam = lambdaparam
        self.weights = None
        self.bias = None
        self.classes = None

    def logistic_function(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        Y = np.zeros((num_samples, num_classes))
        for i, c in enumerate(self.classes):
            Y[:, i] = (y == c).astype(float)

        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros(num_classes)

        for iteration in range(self.iterations):
            weightsum = np.dot(X, self.weights) + self.bias

            predictions = self.logistic_function(weightsum)

            error = Y - predictions

            weightsgradient = (np.dot(X.T, error) - self.lambdaparam * self.weights) / num_samples
            biasgradient = np.mean(error, axis=0)

            self.weights += self.alpha * weightsgradient
            self.bias += self.alpha * biasgradient

    def predict_probability(self, X):
        probs = self.logistic_function(np.dot(X, self.weights) + self.bias)
        return probs / (probs.sum(axis=1, keepdims=True) + 1e-6)

    def predict(self, X):
        return self.classes[self.predict_probability(X).argmax(axis=1)]
    

# No Regularization
class LogisticRegressionSimplified:
    def __init__(self, iterations=1000, alpha=0.01):
        self.iterations = iterations
        self.alpha = alpha
        self.weights = None
        self.classes = None

    def logistic_function(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        Y = np.zeros((num_samples, num_classes))
        for i, c in enumerate(self.classes):
            Y[:, i] = (y == c).astype(float)

        self.weights = np.zeros((num_features, num_classes))

        for iteration in range(self.iterations):
            weightsum = np.dot(X, self.weights)

            predictions = self.logistic_function(weightsum)

            error = Y - predictions

            gradient = np.dot(X.T, error)

            self.weights += self.alpha * gradient

    def predict_probability(self, X):
        probs = self.logistic_function(np.dot(X, self.weights))
        return probs / (probs.sum(axis=1, keepdims=True) + 1e-6)

    def predict(self, X):
        return self.classes[self.predict_probability(X).argmax(axis=1)]