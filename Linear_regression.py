import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

np.random.seed(0)


class LinearRegression:
    def __init__(self, max_iter=1000, learning_rate=1e-3, regularization=False, lambda_=0):
        """
        Parameters
        ----------
        max_iter : int
            The maximum number of iterations the training loop runs for, default 1000
        learning_rate : float
            The learning rate, default 0.001
        regularization : bool
            Whether to use regularization, default False
        lambda_ : float
            The regularization parameter, default = 0
        """

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.parameters = None

        self.regularization = regularization
        self.lambda_ = lambda_

    def forward(self, X):
        return np.dot(X, self.parameters)

    def loss_function(self, yTrue, yPred):
        reg_term = 0
        if self.regularization:
            reg_term = self.lambda_ * np.sum(np.square(self.parameters))

        return np.mean(np.square(yTrue - yPred)) + reg_term

    def gradient(self, X, yTrue, yPred):

        m = yTrue.shape[0]
        dW = (-2 * np.dot(X.T, (yPred - yTrue)) / m)
        if self.regularization:
            dW += 2 * self.lambda_ * self.parameters
        return dW

    def fit(self, xTrain, yTrain):
        """
        Fits the model on the given labelled input data

        Parameters
        ----------
        xTrain : numpy array with shape (m, num_features) where m is the number of training examples
            The training set
        yTrain : numpy ndarray with shape (m,) where m in the number of training examples
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
                print(f"epoch: {epoch}, loss: {_loss:.5f}")

    def predict(self, xTest):
        """
        Parameters
        ----------
        xTest : numpy array
            The input data

        Returns
        -------
        output : numpy array
            The predicted values
        """
        XTest = np.insert(xTest, 0, 1, axis=1)
        yTestPred = self.forward(XTest)
        return yTestPred


x, y = datasets.make_regression(n_samples=100, n_features=1, noise=5)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

n_iter = 300
learning_rate = 0.01

model = LinearRegression(n_iter, learning_rate)
model.fit(x_train, y_train)

plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, model.predict(x), 'r', label='Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points and Fitted Line')
plt.legend()
plt.show()
