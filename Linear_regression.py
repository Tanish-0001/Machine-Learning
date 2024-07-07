import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self, max_iter, eta):
        self.max_iter = max_iter
        self.learning_rate = eta
        self.parameters = None

    @staticmethod
    def forward(X, Theta):
        return np.dot(X, Theta)

    @staticmethod
    def loss_function(yTrue, yPred):
        return np.mean(np.square(yTrue - yPred))

    @staticmethod
    def gradient(X, yTrue, yPred):
        m = yTrue.shape[0]
        dW = -2 * np.dot(X.T, (yPred - yTrue)) / m
        return dW

    def fit(self, xTrain, yTrain):
        self.parameters = np.random.rand(xTrain.shape[1] + 1) if len(xTrain.shape) > 1 else np.random.rand(2)
        if len(xTrain.shape) == 1:
            X_train = np.c_[np.ones(xTrain.shape[0], dtype=xTrain.dtype), xTrain]
        else:
            X_train = np.insert(xTrain, 0, 1, axis=1)

        for epoch in range(self.max_iter):
            Y_pred = self.forward(X_train, self.parameters)
            _loss = self.loss_function(Y_pred, yTrain)
            dw = self.gradient(X_train, Y_pred, yTrain)
            self.parameters -= self.learning_rate * dw

            if epoch % (self.max_iter / 10) == 0:
                print(f"epoch: {epoch}, loss: {_loss}")

    def predict(self, xTest):
        if len(xTest.shape) == 1:
            XTest = np.c_[np.ones(xTest.shape[0], dtype=xTest.dtype), xTest]
        else:
            XTest = np.insert(xTest, 0, 1, axis=1)
        yTestPred = self.forward(XTest, self.parameters)
        return yTestPred


x = np.array([i for i in range(10, 61)], dtype=np.float64)
y = np.array([round(50 * np.log10(x)) for x in range(10, 61)], dtype=np.float64)

x_mean, x_std = np.mean(x), np.std(x)
y_mean, y_std = np.mean(y), np.std(y)

x_norm = (x - np.mean(x)) / np.std(x)
y_norm = (y - np.mean(y)) / np.std(y)

x_train, x_test, y_train, y_test = train_test_split(x_norm, y_norm, test_size=0.2, random_state=42)

n_iter = 100
learning_rate = 0.01

model = LinearRegression(n_iter, learning_rate)
model.fit(x_train, y_train)

params = model.parameters
w = params[1] * (y_std / x_std)
b = params[0] * y_std + y_mean - params[1] * (x_mean * y_std / x_std)

# print("Final weights:", w)
# print("Final bias:", b)

plt.scatter(x, y, color='blue', label='Data points')
x_curve = np.linspace(1, 61, 1000)
plt.plot(x_curve, w * x_curve + b, color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data Points and Line of Best Fit')
plt.legend()
plt.show()
