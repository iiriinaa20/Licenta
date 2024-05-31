import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

        # Mapping class names to class indices starting from 0
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(os.listdir(root)))}

    def __getitem__(self, index):
        # Override the original method to return the correct label
        path, target = self.samples[index]
        sample = self.loader(path)
        # image = cv2.imread(path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # sample = image
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        
      
        # Convert class name to index
        target = self.class_to_idx[os.path.basename(os.path.dirname(path))]
        return sample, target
    
    def display_first_image(self):
        # Display the first image
        if len(self.samples) > 0:
            path, _ = self.samples[0]
            image = Image.open(path)
            image.show()
        else:
            print("No images found in the dataset.")
    
class CNN(nn.Module):
    def __init__(self, num_classes):
        #super(CNN, self).__init__()
        # # self.conv1 = nn.Conv2d(1, 32, kernel_size=3,stride=1, padding=1)
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3,stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.fc1 = nn.Linear(128 * 14 * 14, 128)
        # self.fc2 = nn.Linear(128, num_classes)  
        # self.fc3 = nn.Linear(256, num_classes)
        
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.bn6 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(256 * 14 * 14, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)


    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
    #     x = F.relu(self.conv2(x))
    #     x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
    #     x = F.relu(self.conv3(x))
    #     x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
    #     x = x.view(-1, 128 * 14 * 14)
    #     # print("Shape before flattening:", x.shape)  # Print the shape before flattening

    #     # # Calculate the correct size for flattening
    #     # x = x.view(x.size(0), -1)  # -1 means the remaining dimensions are flattened

    #     # print("Shape after flattening:", x.shape)  # Print the shape after flattening
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     # x = self.fc3(x)
    #     return x
        
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool(x)
        x = self.dropout2(x)

        x = self.bn7(F.relu(self.conv7(x)))
        x = self.pool2(x)
        x = self.dropout3(x)

        x = x.view(-1, 256 * 14 * 14)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    return model
    
def train(model, train_loader, optimizer, criterion, device, epoch, writer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(f"Output batch size: {output.size(0)}, Target batch size: {target.size(0)}")
        # print(f"Output shape: {output.shape}, Target shape: {target.shape}")      
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        
        correct += predicted.eq(target).sum().item()
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / total
    
    train_precision = precision_score(all_targets, all_predictions, average='weighted')
    train_recall = recall_score(all_targets, all_predictions, average='weighted')
    train_f1 = f1_score(all_targets, all_predictions, average='weighted')

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Precision/train', train_precision, epoch)
    writer.add_scalar('Recall/train', train_recall, epoch)
    writer.add_scalar('F1_Score/train', train_f1, epoch)

    return train_loss, train_acc

def validate(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_acc = 100. * correct / total
    
    val_precision = precision_score(all_targets, all_predictions, average='weighted')
    val_recall = recall_score(all_targets, all_predictions, average='weighted')
    val_f1 = f1_score(all_targets, all_predictions, average='weighted')

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Precision/val', val_precision, epoch)
    writer.add_scalar('Recall/val', val_recall, epoch)
    writer.add_scalar('F1_Score/val', val_f1, epoch)

    return val_loss, val_acc

def test(model, test_loader, device):
    model.eval()
    correct = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())

    accuracy = correct / len(test_loader.dataset)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')

    return accuracy, precision, recall, f1

# Define transformations
print(f"Defining transformer")
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    # transforms.ToPILImage(),
    # transforms.Resize((128, 128)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])
print(f"Defined transformer")

# Define the path to your test data folder
# test_data_dir = 'test'
test_data_dir_main = "doubleA"
test_data_dir = "C:\\Users\\mihae\\Desktop\\ooo\\"+test_data_dir_main
model_path = test_data_dir_main+".pth"
# Create a custom dataset
print(f"Processing input")
dataset = CustomImageFolder(test_data_dir, transform=transform)
print(f"Processed input")
# dataset.display_first_image()
# Calculate sizes for train, validation, and test sets
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

print(f"Total Dataset Size: {total_size}")
print(f"Train Dataset Size: {train_size}")
print(f"Validation Dataset Size: {val_size}")
print(f"Test Dataset Size: {test_size}")

# Split the dataset manually
train_data, val_data, test_data = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size], generator=torch.Generator()
    # .manual_seed(42)
)

# Define dataloaders for train, validation, and test sets with drop_last=True
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128)
test_loader = DataLoader(test_data, batch_size=128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

num_classes = len(dataset.classes)
model = CNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

writer = SummaryWriter(log_dir='./logs')

# print(f'Loading Model')
# model.load_state_dict(torch.load('./nn_models/best_modelTest.pth'))
# print(f'Loaded Model')
epochs = 20
best_val_loss = float('inf')
stop_val = 1e-3
# Main training loop
if epochs > 0:
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch, writer)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%')
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model,model_path)
            # torch.save(model.state_dict(), 'best_modelTest2.pth')
        if train_loss < stop_val:
            break

print('Loading Model')
model = load_model(model, model_path)
# model.load_state_dict(torch.load('./nn_models/best_modelTest.pth'))
print("Starting Testing")
test_accuracy, test_precision, test_recall, test_f1 = test(model, test_loader, device)

print(f'Test Accuracy: {test_accuracy}')
print(f'Test Precision: {test_precision}')
print(f'Test Recall: {test_recall}')
print(f'Test F1: {test_f1}')
