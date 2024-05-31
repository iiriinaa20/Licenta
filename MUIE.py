import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import os
import detect_face

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Custom dataset
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.image_paths = []
        self.labels = []
        self.labels_unique = []

        for label, cls in enumerate(self.classes):
            cls_folder = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Data transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader
dataset = FaceDataset(root_dir=r"C:\Users\mihae\Desktop\ooo", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

num_classes = len(os.listdir(r"C:\Users\mihae\Desktop\ooo"))
model = SimpleCNN(num_classes=num_classes).to(device)

# Training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 10 == 9:  # Print every 10 batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save the model
torch.save(model.state_dict(), 'face_recognition_model.pth')

# Inference function
def predict(image_path, model, transform):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    if height != 112 and width != 112:
        image = detect_face.detect_and_return_face(image_path)
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

# Load the model for inference
model.load_state_dict(torch.load('face_recognition_model.pth'))

test_image_path = r"C:\Users\mihae\Desktop\meg.jpg"
prediction = predict(test_image_path, model, transform)
print(f'Predicted class: {prediction}')
cv2.imshow("Image4", cv2.resize( cv2.imread(test_image_path), (400, 400), interpolation=cv2.INTER_AREA))

test_image_path = r"C:\Users\mihae\Desktop\k.jpg"
prediction = predict(test_image_path, model, transform)
print(f'Predicted class: {prediction}')
#cv2.imshow("Image1", cv2.imread(test_image_path))

cv2.waitKey(0)
cv2.destroyAllWindows() 
