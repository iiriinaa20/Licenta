import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(os.listdir(root)))}

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.class_to_idx[os.path.basename(os.path.dirname(path))]
        return sample, target

    def display_first_image(self):
        if len(self.samples) > 0:
            path, _ = self.samples[0]
            image = Image.open(path)
            image.show()
        else:
            print("No images found in the dataset.")

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = torch.nn.Linear(3 * 28 * 28, num_classes, bias=True)  # Adjusted this line to take num_classes
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   
        out = self.fc(out)
        return out

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    return val_loss, accuracy

def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Changed to Grayscale(num_output_channels=3)
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

test_data_dir = "C:\\Users\\mihae\\Desktop\\test1"
dataset = CustomImageFolder(test_data_dir, transform=transform)

total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_data, val_data, test_data = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(dataset.classes)
model = CNN(num_classes).to(device)  # Passed num_classes to CNN
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

epochs = 100
best_val_loss = float('inf')
for epoch in range(1, epochs + 1):
    train(model, train_loader, optimizer, criterion, device)
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    print(f'Epoch: {epoch}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

model.load_state_dict(torch.load('best_model.pth'))
test_accuracy = test(model, test_loader, device)
print(f'Test Accuracy: {test_accuracy}')

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
