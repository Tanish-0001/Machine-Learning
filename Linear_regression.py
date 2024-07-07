import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LinearRegression:
    """
    A simple linear regression model

    ...

    Attributes
    ----------
    max_iter : int
        Maximum number of iterations used for gradient descent
    learning_rate : float
        Learning rate

    Methods
    -------
    forward(X)
        Computes the forward pass

    loss_function(yTrue, yPred)
        Computes the mean squared error

    gradient(X, yTrue, yPred)
        Computes backward pass to update model parameters

    fit(xTrain, yTrain)
        Fits model to training data

    predict(xTest)
        Makes a prediction using the trained model
    """

    def __init__(self, max_iter, learning_rate):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.parameters = None

    def forward(self, X):
        return np.dot(X, self.parameters)

    @staticmethod
    def loss_function(yTrue, yPred):
        return np.mean(np.square(yTrue - yPred))

    @staticmethod
    def gradient(X, yTrue, yPred):
        m = yTrue.shape[0]
        dW = -2 * np.dot(X.T, (yPred - yTrue)) / m
        return dW

    def fit(self, xTrain, yTrain):
        """
        Fits the model on the given labelled input data

        Parameters
        ----------
        xTrain : numpy ndarray with shape (m, num_features) where m is the number of training examples
            The training set
        yTrain: numpy ndarray with shape (m,) where m in the number of training examples
            The corresponding labels
        """

        self.parameters = np.random.rand(xTrain.shape[1] + 1)
        X_train = np.insert(xTrain, 0, 1, axis=1)

        for epoch in range(self.max_iter):
            Y_pred = self.forward(X_train)
            _loss = self.loss_function(Y_pred, yTrain)
            dw = self.gradient(X_train, Y_pred, yTrain)
            self.parameters -= self.learning_rate * dw

            if epoch % (self.max_iter / 10) == 0:
                print(f"epoch: {epoch}, loss: {_loss}")

    def predict(self, xTest):
        XTest = np.insert(xTest, 0, 1, axis=1)
        yTestPred = self.forward(XTest)
        return yTestPred


x = np.array([[i] for i in range(10, 61)], dtype=np.float64)
y = np.array([round(50 * np.log10(x)) for x in range(10, 61)], dtype=np.float64)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_mean, x_std = np.mean(x_train), np.std(x_train)
y_mean, y_std = np.mean(y_train), np.std(y_train)

x_norm = (x_train - x_mean) / x_std
y_norm = (y_train - y_mean) / y_std

n_iter = 100
learning_rate = 0.01

model = LinearRegression(n_iter, learning_rate)
model.fit(x_norm, y_norm)

params = model.parameters
w = params[1] * (y_std / x_std)
b = params[0] * y_std + y_mean - params[1] * (x_mean * y_std / x_std)

# print("Final weights:", w)
# print("Final bias:", b)

plt.scatter(x, y, color='blue', label='Data points')
x_curve = np.linspace(10, 61, 1000).reshape(-1, 1)
plt.plot(x_curve, w * x_curve + b, color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data Points and Line of Best Fit')
plt.legend()
plt.show()
