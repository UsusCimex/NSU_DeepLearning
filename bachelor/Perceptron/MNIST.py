from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from perceptrons.pyTorchPerceptron import MultilayerPerceptron, PerceptronEnsemble
from PIL import Image, ImageOps

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_data = datasets.MNIST(root='./data_generation', train=True, download=True, transform=transform)
    
    train_loader = DataLoader(mnist_data, batch_size=64, shuffle=True)

    # layers_sizes = [28 * 28, 128, 64, 10] 
    # learning_rate = 0.001
    # activations = [nn.ReLU(), nn.ReLU(), nn.Sigmoid()]
    # model = MultilayerPerceptron(layers_sizes, learning_rate, activations)
    # model.fit(train_loader, epochs=5)

    input_size = 28 * 28
    output_size = 10
    n_perceptrons = 7
    model = PerceptronEnsemble(input_size, output_size, n_perceptrons)
    model.fit(train_loader, epochs=6, learning_rate=0.001)

    # Тестирование на примере изображения
    image_path = './mnist_test.png'
    while (input("Retry? (y/n): ") != "n"):
        image = Image.open(image_path).convert('L')
        image = ImageOps.invert(image)
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = transform(image).unsqueeze(0)

        prediction = model.predict(image)
        print("Predicted digit:", prediction)

if __name__ == "__main__":
    main()