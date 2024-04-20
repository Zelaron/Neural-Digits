import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

# Ask the user for the program mode
while True:
    try:
        programMode = int(input("Enter the program mode (0: train, 1: test, 2: infer): "))
        if programMode in [0, 1, 2]:
            break
        else:
            print("Please enter 0, 1, or 2.")
    except ValueError:
        print("Please enter a valid integer.")

# File names
model_file = 'mnist_model.pth'
data_directory = './data'

# Process an image file to map each pixel to a floating point value between 0 and 1
def process_image(file_path):
    image = Image.open(file_path)
    image = image.convert('L')
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
    data = np.asarray(image)
    # Check if the image is likely to have a white background
    if data.sum() > 127.5 * 28 * 28:
        data = 255 - data  # Invert the pixel values
    
    normalized_data = data / 255.0
    return normalized_data.flatten()

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if programMode == 0:
    # Initialize the neural network model
    model = SimpleNN()

    # Check if the model file exists
    model_exists = os.path.exists(model_file)

    # Ask if the user wants to retrain from scratch only if a model file exists
    retrain_from_scratch = False
    if model_exists:
        while True:
            try:
                user_input = input("Retrain from scratch? (1: Yes, 0: No): ")
                retrain_from_scratch = int(user_input.strip())
                if retrain_from_scratch in [0, 1]:
                    retrain_from_scratch = bool(retrain_from_scratch)
                    break
                else:
                    print("Please enter 0 or 1.")
            except ValueError:
                print("Please enter a valid integer.")

        if retrain_from_scratch:
            os.remove(model_file)
            print(f"Deleted existing model file: {model_file}")

            # Delete data directory if retraining from scratch
            if os.path.exists(data_directory):
                shutil.rmtree(data_directory)
                print(f"Deleted existing data directory: {data_directory}")

    # Load the model if it exists and the user does not want to retrain from scratch
    if model_exists and not retrain_from_scratch:
        model.load_state_dict(torch.load(model_file))
        print("Loaded saved model for further training.")
    else:
        print("No saved model found, starting training from scratch.")

    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Define data transformations and load training dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(root=data_directory, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop
    total_epochs = 20 # Define total number of epochs
    for epoch in range(total_epochs):  # Adjust the number of epochs if needed
        epoch_loss = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{total_epochs}")

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=epoch_loss / len(train_loader))

    # Save the trained model
    torch.save(model.state_dict(), model_file)

elif programMode == 1:
    # Initialize and load the trained model
    model = SimpleNN()
    model.load_state_dict(torch.load(model_file))
    model.eval()  # Set the model to evaluation mode

    # Prepare the test dataset
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = MNIST(root=data_directory, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate the model on the test dataset
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients for testing
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and print the accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')

elif programMode == 2:
    # Load the trained model
    model = SimpleNN()
    model.load_state_dict(torch.load(model_file))
    model.eval()  # Set the model to evaluation mode

    # Run inference
    example_input = process_image('digit.png')
    example_input_tensor = torch.from_numpy(np.array([example_input], dtype=np.float32))
    output = model(example_input_tensor)
    softmax = nn.Softmax(dim=1)
    probabilities = softmax(output)
    probabilities_list = [f"{prob:.11f}" for prob in probabilities.detach().numpy().flatten()]
    print("Probabilities: [" + ', '.join(probabilities_list) + "]")
    predicted = torch.max(probabilities, 1)[1]
    print(f"The most probable digit is {predicted.item()}.")