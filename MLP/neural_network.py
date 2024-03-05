import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def forward(self, X, change_value=True):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)

        if change_value:
            self.Z1, self.A1, self.Z2, self.A2 = Z1, A1, Z2, A2

        return A2

    def backprop(self, X, y, learning_rate):
        m = y.shape[0]
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, self.W2.T) * (1 - np.power(self.A1, 2))
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, iterations=10000, learning_rate=1.2, visualization_func=None):
        loss_history = []
        for i in range(iterations):
            y_pred = self.forward(X)
            loss = self.loss(y, y_pred)
            loss_history.append(loss)

            if i % (iterations//5) == 0:
                print(f"Iteration {i}, Loss: {loss}")
                if visualization_func is not None:
                    visualization_func(self, X, y, i, loss)

            self.backprop(X, y, learning_rate)
        return loss_history
    
    def predict(self, X, change_value=True):
        """
        Функция для выполнения прогнозирования на основе обученной модели.
        :param X: Массив признаков
        """
        y_pred = self.forward(X, change_value)
        return (y_pred > 0.5).astype(int)
