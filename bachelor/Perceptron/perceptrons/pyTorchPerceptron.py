import torch
from torch import nn
from torch import optim

class Perceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_size, output_size) 

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class PerceptronEnsemble(nn.Module):
    def __init__(self, input_size, output_size, n_perceptrons):
        super(PerceptronEnsemble, self).__init__()
        self.perceptrons = nn.ModuleList([Perceptron(input_size, output_size) for _ in range(n_perceptrons)])

    def forward(self, x):
        outputs = torch.stack([torch.softmax(p(x), dim=1) for p in self.perceptrons], dim=1)
        return outputs.mean(dim=1)

    def predict(self, x):
        with torch.no_grad():
            outputs = self(x)
            return outputs.argmax(dim=1)

    def fit(self, data_loader, epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for X, y in data_loader:
                optimizer.zero_grad()
                outputs = self(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}')

class MultilayerPerceptron(nn.Module):
    def __init__(self, layers_sizes, learning_rate, activations):
        super(MultilayerPerceptron, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_sizes) - 1):
            self.layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))

        if len(activations) != len(layers_sizes) - 1:
            raise ValueError("The number of activation functions must be equal to the number of layers - 1")

        self.activations = activations
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = activation(layer(x))
        x = self.layers[-1](x)
        return x

    def fit(self, train_loader, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for X, y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.forward(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch % 100) == 99:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')

    def predict(self, X):
        with torch.no_grad():
            outputs = self.forward(X)
            _, predictions = torch.max(outputs, 1)
            return predictions.item()