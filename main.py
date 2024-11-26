import os
import cv2
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ------------------ Frame Extraction ------------------
DATA_DIR = "./data/UCF-101"
FRAME_DIR = "./data/frames"

def extract_frames(video_dir, frame_dir):
    """Extract frames from all videos in UCF-101 dataset."""
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    for class_name in os.listdir(video_dir):
        class_path = os.path.join(video_dir, class_name)
        if os.path.isdir(class_path):
            print(f"Processing class: {class_name}")
            class_frame_dir = os.path.join(frame_dir, class_name)
            if not os.path.exists(class_frame_dir):
                os.makedirs(class_frame_dir)
            for video in os.listdir(class_path):
                video_path = os.path.join(class_path, video)
                video_name = os.path.splitext(video)[0]
                video_frame_dir = os.path.join(class_frame_dir, video_name)
                if not os.path.exists(video_frame_dir):
                    os.makedirs(video_frame_dir)
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_path = os.path.join(video_frame_dir, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_count += 1
                cap.release()
                print(f"Extracted {frame_count} frames from {video_name}")

# ------------------ Dataset Class ------------------
class UCF101Dataset(Dataset):
    def __init__(self, frame_dir, max_classes=None, max_videos_per_class=None, transform=None, max_frames=16):
        self.frame_dir = frame_dir
        self.transform = transform
        self.max_frames = max_frames
        self.samples = []
        self.classes = sorted(os.listdir(frame_dir))
        if max_classes:
            random.seed()  # Ensure randomness each time
            random.shuffle(self.classes)
            self.classes = self.classes[:max_classes]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for cls in self.classes:
            class_path = os.path.join(frame_dir, cls)
            if os.path.isdir(class_path):
                videos = sorted(os.listdir(class_path))
                if max_videos_per_class:
                    videos = videos[:max_videos_per_class]
                for video in videos:
                    video_path = os.path.join(class_path, video)
                    if os.path.isdir(video_path):
                        frame_paths = sorted(os.listdir(video_path))
                        frame_paths = [os.path.join(video_path, frame) for frame in frame_paths]
                        for frame_path in frame_paths:
                            self.samples.append((frame_path, self.class_to_idx[cls]))
        print(f"Randomly Selected Classes: {self.classes}")
        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, label = self.samples[idx]
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------ CNN Model ------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet = resnet18(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# ------------------ GNN Model ------------------
class CNNGNNModel(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(CNNGNNModel, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet = resnet18(weights=weights)
        resnet_feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove classification head

        self.conv1 = GCNConv(resnet_feature_dim, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x

# ------------------ Dataset Split ------------------
def create_shared_dataset(frame_dir, max_classes, max_videos_per_class, test_split, transform):
    random.seed(42)
    torch.manual_seed(42)
    dataset = UCF101Dataset(frame_dir, max_classes=max_classes, max_videos_per_class=max_videos_per_class, transform=transform)
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    print(f"Shared Train set: {len(train_dataset)} samples, Test set: {len(test_dataset)} samples.")
    return train_dataset, test_dataset

# ------------------ CNN Training Script ------------------
def train_cnn_model(train_dataset, test_dataset, num_classes):
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"CNN Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    return model, test_loader

# ------------------ GNN Training Script ------------------
def train_gnn_model(train_dataset, test_dataset, num_classes):
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNGNNModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)
            features = model.resnet(frames)
            num_nodes = features.size(0)
            edge_index = torch.tensor(
                [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
                dtype=torch.long
            ).t().to(device)

            if edge_index.numel() == 0:
                continue

            outputs = model(features, edge_index)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"GNN Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    print("GNN Training complete.")
    return model, test_loader

# ------------------ Metrics ------------------
def evaluate_model(model, test_loader, num_classes, gnn=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(device), labels.to(device)

            if gnn:
                # For GNN, extract features and create edge_index
                features = model.resnet(frames)
                num_nodes = features.size(0)
                edge_index = torch.tensor(
                    [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
                    dtype=torch.long
                ).t().to(device)

                if edge_index.numel() == 0:  # Skip if edge_index is empty
                    print("Skipping batch with no edges...")
                    continue

                outputs = model(features, edge_index)
            else:
                # For CNN
                outputs = model(frames)

            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {acc * 100:.2f}%, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    return acc, f1, precision, recall

# ------------------ Main Execution ------------------
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    max_classes = 2
    max_videos_per_class = 2
    test_split = 0.2

    print("Creating shared dataset...")
    train_dataset, test_dataset = create_shared_dataset(FRAME_DIR, max_classes, max_videos_per_class, test_split, transform)

    print("\nTraining CNN Model...")
    cnn_model, cnn_test_loader = train_cnn_model(train_dataset, test_dataset, num_classes=max_classes)
    print("Evaluating CNN Model...")
    evaluate_model(cnn_model, cnn_test_loader, num_classes=max_classes)

    print("\nTraining GNN Model...")
    gnn_model, gnn_test_loader = train_gnn_model(train_dataset, test_dataset, num_classes=max_classes)
    print("Evaluating GNN Model...")
    evaluate_model(gnn_model, gnn_test_loader, num_classes=max_classes)

