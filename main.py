import os
import re
import cv2
import glob
import json
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from matplotlib import image
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur
from transformers import AutoImageProcessor, AutoModelForImageClassification

def read_image_file(path):
    image_paths = glob.glob(path + "/*.png")
    image_data = []

    # Define the transformation to convert images to PyTorch tensors
    to_tensor_transform = transforms.ToTensor()

    for image_path in image_paths:
        img_name = os.path.basename(image_path)
        # Define a regular expression pattern to match the label number
        pattern = re.compile(r'label_(\d+)')
        # Use the findall method to extract all matches
        matches = pattern.findall(img_name)
        label_number = int(matches[0])

        # Read the image using OpenCV
        img = cv2.imread(image_path)

        if img is not None:
            # Apply the ToTensor transformation to convert the image to a PyTorch tensor
            img_tensor = to_tensor_transform(img)
            image_data.append((img_tensor, label_number))
        else:
            print(f"Error reading image: {image_path}")

    return image_data

training_data  = read_image_file('path-to-dataset-train')
test_data = read_image_file('path-to-dataset-test')

transform = transforms.Compose([
    transforms.RandomApply([transforms.RandomRotation(degrees=(-45, 45))], p=0.7),  # Rotate the images by random angles to simulate variations in the orientation of handwritten charecter.
    transforms.RandomApply([transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x))], p=0.3), # Adds noise to images, simulating pixel-level variations.
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3), # Simulates out-of-focus or blurry images.
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.4) # Adjust the brightness and contrast of the images randomly to capture variations in lighting conditions.
])

augmented_training_data = []

for img, label in training_data:
  for i in range(0,3):
    augmented_image = transform(img)
    augmented_training_data.append((augmented_image,label))

training_data2 = training_data + augmented_training_data

"""### 2.2.3. Displaying some samples of the data"""

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    if img.shape[0] == 1:
        img = img.squeeze()
    else:
        img = img.permute(1, 2, 0)
    figure.add_subplot(rows, cols, i)
    plt.title("Class:" + str(label))
    plt.axis("off")
    plt.imshow(img, cmap="gray")
plt.show()

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data2), size=(1,)).item()
    img, label = training_data2[sample_idx]
    if img.shape[0] == 1:
        img = img.squeeze()
    else:
        img = img.permute(1, 2, 0)
    figure.add_subplot(rows, cols, i)
    plt.title("Class:" + str(label))
    plt.axis("off")
    plt.imshow(img, cmap="gray")
plt.show()


class ArabicLettersCNN(nn.Module):
    def __init__(self):
            super(ArabicLettersCNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
            # Max pooling layers
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            # Dropout layers
            self.dropout1 = nn.Dropout(0.2)
            self.dropout2 = nn.Dropout(0.4)
            # Fully-connected layers
            self.fc1 = nn.Linear(128 * 2 * 2, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 29)

    def forward(self, x):
            # Define the forward pass using the layers with shape printing
            x = self.pool(F.relu(self.conv1(x))) # output shape 16*16*16
            x = self.pool(F.relu(self.conv2(x))) # output shape 8*8*32
            x = self.dropout1(x)
            x = self.pool(F.relu(self.conv3(x))) # output shape 4*4*64
            x = self.pool(F.relu(self.conv4(x))) # output shape 2*2*128
            x = self.dropout2(x)
            x = x.view(x.size(0), -1) # output shape 2*2*128 = 512
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

# Initializing the model, loss function, and optimizer
model = ArabicLettersCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
num_epochs = 16

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


for epoch in range(num_epochs):
    # Training loop
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_dataloader)
    epoch_train_accuracy = 100 * correct_train / total_train

    # Test loop
    model.eval()
    with torch.no_grad():
      test_loss = 0.0
      correct_test = 0
      total_test = 0
      for data in test_dataloader:
          inputs, labels = data[0].to(device), data[1].to(device)

          outputs = model(inputs)

          loss = criterion(outputs, labels)

          test_loss += loss.item()

          _, predicted = torch.max(outputs.data, 1)
          total_test += labels.size(0)
          correct_test += (predicted == labels).sum().item()

      # calculate average testing loss
      average_test_loss = test_loss / len(test_dataloader)
      epoch_test_accuracy = 100 * correct_test // total_test

    # Append metrics to lists for plotting
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)
    test_losses.append(average_test_loss)
    test_accuracies.append(epoch_test_accuracy)

    # Print and/or log the metrics for each epoch
    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Training Loss: {epoch_train_loss:.4f}, '
          f'Training Accuracy: {epoch_train_accuracy:.4f}, '
          f'Testing Loss: {average_test_loss:.4f}, '
          f'Testing Accuracy: {epoch_test_accuracy:.4f}')

# Plotting
plt.figure(figsize=(18, 4))

# Plot Training Loss
plt.subplot(1, 4, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Training  Accuracy
plt.subplot(1, 4, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Testing Loss
plt.subplot(1, 4, 3)
plt.plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Testing Accuracy
plt.subplot(1, 4, 4)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_dataloader = DataLoader(training_data2, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

# Initializing the model, loss function, and optimizer
model = ArabicLettersCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
num_epochs = 20

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


for epoch in range(num_epochs):
    # Training loop
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_dataloader)
    epoch_train_accuracy = 100 * correct_train / total_train

    # Test loop
    model.eval()
    with torch.no_grad():
      test_loss = 0.0
      correct_test = 0
      total_test = 0
      for data in test_dataloader:
          inputs, labels = data[0].to(device), data[1].to(device)

          outputs = model(inputs)

          loss = criterion(outputs, labels)

          test_loss += loss.item()

          _, predicted = torch.max(outputs.data, 1)
          total_test += labels.size(0)
          correct_test += (predicted == labels).sum().item()

      # calculate average testing loss
      average_test_loss = test_loss / len(test_dataloader)
      epoch_test_accuracy = 100 * correct_test // total_test

    # Append metrics to lists for plotting
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)
    test_losses.append(average_test_loss)
    test_accuracies.append(epoch_test_accuracy)

    # Print and/or log the metrics for each epoch
    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Training Loss: {epoch_train_loss:.4f}, '
          f'Training Accuracy: {epoch_train_accuracy:.4f}, '
          f'Testing Loss: {average_test_loss:.4f}, '
          f'Testing Accuracy: {epoch_test_accuracy:.4f}')

# Plotting
plt.figure(figsize=(18, 4))

# Plot Training Loss
plt.subplot(1, 4, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Training  Accuracy
plt.subplot(1, 4, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Testing Loss
plt.subplot(1, 4, 3)
plt.plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Testing Accuracy
plt.subplot(1, 4, 4)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_dataloader = DataLoader(training_data2, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

# Initializing the model, loss function, and optimizer
model = models.resnext50_32x4d()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 29)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
num_epochs = 20

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


for epoch in range(num_epochs):
    # Training loop
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_dataloader)
    epoch_train_accuracy = 100 * correct_train / total_train

    # Test loop
    model.eval()
    with torch.no_grad():
      test_loss = 0.0
      correct_test = 0
      total_test = 0
      for data in test_dataloader:
          inputs, labels = data[0].to(device), data[1].to(device)

          outputs = model(inputs)

          loss = criterion(outputs, labels)

          test_loss += loss.item()

          _, predicted = torch.max(outputs.data, 1)
          total_test += labels.size(0)
          correct_test += (predicted == labels).sum().item()

      # calculate average testing loss
      average_test_loss = test_loss / len(test_dataloader)
      epoch_test_accuracy = 100 * correct_test // total_test

    # Append metrics to lists for plotting
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)
    test_losses.append(average_test_loss)
    test_accuracies.append(epoch_test_accuracy)

    # Print and/or log the metrics for each epoch
    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Training Loss: {epoch_train_loss:.4f}, '
          f'Training Accuracy: {epoch_train_accuracy:.4f}, '
          f'Testing Loss: {average_test_loss:.4f}, '
          f'Testing Accuracy: {epoch_test_accuracy:.4f}')

# Plotting
plt.figure(figsize=(18, 4))

# Plot Training Loss
plt.subplot(1, 4, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Training  Accuracy
plt.subplot(1, 4, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Testing Loss
plt.subplot(1, 4, 3)
plt.plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Testing Accuracy
plt.subplot(1, 4, 4)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_dataloader = DataLoader(training_data2, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

# Initializing the model, loss function, and optimizer
model = models.resnext50_32x4d(weights='IMAGENET1K_V2')

num_features = model.fc.in_features

hidden_layers = [nn.Linear(num_features, 1024),
                 nn.ReLU(),
                 nn.Dropout(0.5)]

fc_layers = [nn.Linear(1024, 512),
             nn.ReLU(),
             nn.Dropout(0.5),
             nn.Linear(512, 256)]

new_classifier = nn.Sequential(*hidden_layers, *fc_layers)

model.fc = new_classifier

model.avgpool2 = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

model.fc2 = torch.nn.Linear(256, 29)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
num_epochs = 12

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


for epoch in range(num_epochs):
    # Training loop
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_dataloader)
    epoch_train_accuracy = 100 * correct_train / total_train

    # Test loop
    model.eval()
    with torch.no_grad():
      test_loss = 0.0
      correct_test = 0
      total_test = 0
      for data in test_dataloader:
          inputs, labels = data[0].to(device), data[1].to(device)

          outputs = model(inputs)

          loss = criterion(outputs, labels)

          test_loss += loss.item()

          _, predicted = torch.max(outputs.data, 1)
          total_test += labels.size(0)
          correct_test += (predicted == labels).sum().item()

      # calculate average testing loss
      average_test_loss = test_loss / len(test_dataloader)
      epoch_test_accuracy = 100 * correct_test // total_test

    # Append metrics to lists for plotting
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)
    test_losses.append(average_test_loss)
    test_accuracies.append(epoch_test_accuracy)

    # Print and/or log the metrics for each epoch
    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Training Loss: {epoch_train_loss:.4f}, '
          f'Training Accuracy: {epoch_train_accuracy:.4f}, '
          f'Testing Loss: {average_test_loss:.4f}, '
          f'Testing Accuracy: {epoch_test_accuracy:.4f}')

# Plotting
plt.figure(figsize=(18, 4))

# Plot Training Loss
plt.subplot(1, 4, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Training  Accuracy
plt.subplot(1, 4, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Testing Loss
plt.subplot(1, 4, 3)
plt.plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Testing Accuracy
plt.subplot(1, 4, 4)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

