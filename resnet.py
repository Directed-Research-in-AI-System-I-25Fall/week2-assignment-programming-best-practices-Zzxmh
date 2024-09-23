import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ResNetForImageClassification, ResNetConfig
from tqdm import tqdm

# Step 1: Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 as expected by ResNet
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the dataset
])

mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

# Step 2: Modify the ResNet model to output 10 classes for MNIST
config = ResNetConfig.from_pretrained("microsoft/resnet-50")
config.num_labels = 10  # MNIST has 10 classes (digits 0-9)
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", config=config,ignore_mismatched_sizes=True)
model.eval()  # Set model to evaluation mode

# Step 3: Initialize variables to track accuracy
correct = 0
total = 0

# Step 4: Run inference on the MNIST test dataset with real-time progress monitoring
with tqdm(total=len(test_loader), desc="Testing Progress") as progress_bar:
    for i, (images, labels) in enumerate(test_loader):
        images = images.repeat(1, 3, 1, 1)  # Repeat 1 channel to 3 channels (RGB)

        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs.logits, 1)  # Get predictions

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Print interim accuracy
        batch_accuracy = (preds == labels).sum().item() / labels.size(0) * 100
        print(f'Batch {i+1}/{len(test_loader)} Accuracy: {batch_accuracy:.2f}%')
        progress_bar.update(1)

# Step 5: Calculate and print final accuracy
accuracy = correct / total * 100
print(f'\nFinal Accuracy of the pretrained ResNet model on the MNIST dataset: {accuracy:.2f}%')
