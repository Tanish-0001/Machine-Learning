import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(1000)


class LogisticRegression:
    def __init__(self, max_iter=1000, eta=0.001):
        self.parameters = None
        self.max_iter = max_iter
        self.learning_rate = eta

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        return self.sigmoid(np.dot(X, self.parameters))

    @staticmethod
    def loss_function(yPred, yTrue):
        epsilon = 1e-15
        yPred = np.clip(yPred, epsilon, 1 - epsilon)  # to prevent log(0) when y_pred == 0
        return -np.mean(yTrue * np.log(yPred) + (1 - yTrue) * np.log(1 - yPred))

    @staticmethod
    def gradient(X, yPred, yTrue):
        m = yTrue.shape[0]
        dW = np.dot((yTrue - yPred), X) / m
        return dW

    def fit(self, xTrain, yTrain):
        self.parameters = np.random.rand(xTrain.shape[1] + 1)
        XTrain = np.insert(xTrain, 0, 1, axis=1)

        for epoch in range(self.max_iter):
            Y_pred = self.forward(XTrain)
            _loss = self.loss_function(Y_pred, yTrain)
            dw = self.gradient(XTrain, Y_pred, yTrain)
            self.parameters += self.learning_rate * dw

            if epoch % (self.max_iter / 10) == 0:
                print(f"epoch: {epoch}, loss: {_loss}")

    def predict(self, xTest):
        XTest = np.insert(xTest, 0, 1, axis=1)
        yTestPred = self.forward(XTest)
        return yTestPred
        

data = load_breast_cancer()
x, y = data['data'], data['target']
x = StandardScaler().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

n_iter = 1000
learning_rate = 0.01

model = LogisticRegression(max_iter=n_iter, eta=learning_rate)
model.fit(x_train, y_train)

y_test_pred = model.predict(x_test)
y_test_pred_class = np.array([1 if i > 0.5 else 0 for i in y_test_pred])  # threshold probability = 0.5
accuracy = np.mean(y_test_pred_class == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix, Precision, Recall, F1 Score
tp = np.sum(np.logical_and(y_test_pred_class == 1, y_test == 1))
tn = np.sum(np.logical_and(y_test_pred_class == 0, y_test == 0))
fp = np.sum(np.logical_and(y_test_pred_class == 1, y_test == 0))
fn = np.sum(np.logical_and(y_test_pred_class == 0, y_test == 1))

print(f"Confusion matrix:\n{np.array([[tp, fn], [fp, tn]])}")

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.2f}, Recall: {recall}, F1: {f1_score:.2f}")
