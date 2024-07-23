import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(0)


class SoftmaxRegression:
    def __init__(self, n_classes, max_iter=1000, eta=1e-3):
        self.max_iter = max_iter
        self.learning_rate = eta
        self.n_classes = n_classes
        self.parameters = None

    @staticmethod
    def cross_entropy_loss(actual, predicted):
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)  # to prevent log(0)
        return -np.mean(np.sum(actual * np.log(predicted), axis=1))

    @staticmethod
    def softmax(X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)

    @staticmethod
    def gradient(X, y, y_pred):
        return np.dot((y_pred - y).T, X) / X.shape[0]

    def forward(self, X):
        return self.softmax(np.dot(X, self.parameters.T))

    def fit(self, X, y):
        self.parameters = np.random.randn(self.n_classes, X.shape[1] + 1)  # each class has its own set of weights and bias
        X = np.insert(X, 0, 1, axis=1)  # add a "constant" feature for the bias term

        for epoch in range(self.max_iter):
            y_pred = self.forward(X)
            loss = self.cross_entropy_loss(y, y_pred)
            dw = self.gradient(X, y, y_pred)
            self.parameters -= self.learning_rate * dw

            if epoch % (self.max_iter // 10) == 0:
                print(f"Epoch: {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        scores = self.forward(X)
        return np.argmax(scores, axis=1)


model = SoftmaxRegression(max_iter=1000, eta=0.001, n_classes=2)

data = load_breast_cancer()
x, y = data['data'], data['target']
x = StandardScaler().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_train = np.eye(2)[y_train]  # converts into one-hot encoding

model.fit(x_train, y_train)

y_test_pred = model.predict(x_test)
accuracy = np.mean(y_test_pred == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

true_positive = 0
false_negative = 0
for pred, true in zip(y_test_pred, y_test):
    if pred == 1 and true == 1:
        true_positive += 1
    elif pred == 0 and true == 1:
        false_negative += 1

print(f'Recall: {true_positive / (true_positive + false_negative):.3f}')
