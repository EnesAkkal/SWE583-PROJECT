import os
import cv2
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.transforms import ToPILImage

# ------------------ Frame Extraction ------------------
DATA_DIR = "./data/UCF-101"  # Path to UCF-101 dataset
FRAME_DIR = "./data/frames"  # Path where extracted frames will be stored
BASE_RESULTS_DIR = "./results"  # Base directory for results

if not os.path.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

# ------------------ Dataset Class ------------------
class UCF101Dataset(Dataset):
    def __init__(self, frame_dir, max_classes=None, max_videos_per_class=None, transform=None, reuse_classes=False):
        self.frame_dir = frame_dir
        self.transform = transform
        self.samples = []
        self.classes_file = os.path.join(BASE_RESULTS_DIR, "selected_classes.txt")

        all_classes = [cls for cls in sorted(os.listdir(frame_dir)) if os.path.isdir(os.path.join(frame_dir, cls))]

        if max_classes:
            reuse = input("Do you want to reuse previously selected classes? (yes/no): ").strip().lower()
            reuse_classes = True if reuse == "yes" else False

            if reuse_classes and os.path.exists(self.classes_file):
                with open(self.classes_file, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
            else:
                random.shuffle(all_classes)
                self.classes = all_classes[:max_classes]
                with open(self.classes_file, "w") as f:
                    for cls in self.classes:
                        f.write(cls + "\n")
        else:
            self.classes = all_classes

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            class_path = os.path.join(frame_dir, cls)
            videos = sorted(os.listdir(class_path))
            if max_videos_per_class:
                videos = videos[:max_videos_per_class]

            for video in videos:
                video_path = os.path.join(class_path, video)
                if not os.path.isdir(video_path):  # Skip files like .DS_Store
                    continue
                for frame in os.listdir(video_path):
                    frame_path = os.path.join(video_path, frame)
                    if not os.path.isfile(frame_path):  # Ensure it's a valid file
                        continue
                    self.samples.append((frame_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, label = self.samples[idx]
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------ ViT Model ------------------
class ActionRecognitionModelViT(nn.Module):
    def __init__(self, num_classes):
        super(ActionRecognitionModelViT, self).__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

# ------------------ Training Script ------------------
def train_model(max_classes=100, max_videos_per_class=1, test_split=0.2, reuse_classes=False, learning_rate=0.01, batch_size=64, epochs=3):
    results_dir = os.path.join(
        BASE_RESULTS_DIR,
        f"vit_classes_{max_classes}_videos_{max_videos_per_class}_lr_{learning_rate}_bs_{batch_size}_epochs_{epochs}"
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = UCF101Dataset(FRAME_DIR, max_classes=max_classes, max_videos_per_class=max_videos_per_class, transform=transform, reuse_classes=reuse_classes)
    print(f"Loaded {len(dataset)} samples from {len(dataset.classes)} classes.")

    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Train set: {len(train_dataset)} samples, Test set: {len(test_dataset)} samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionRecognitionModelViT(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(test_loader)
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Save loss and accuracy plots
    plt.figure()
    plt.plot(range(1, epochs+1), history["train_loss"], label="Training Loss")
    plt.plot(range(1, epochs+1), history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir, "train_val_loss.png"))

    plt.figure()
    plt.plot(range(1, epochs+1), history["train_acc"], label="Training Accuracy")
    plt.plot(range(1, epochs+1), history["val_acc"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir, "train_val_accuracy.png"))

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))

    print("Training complete. Metrics, visualizations, and confusion matrix saved.")

    return model, test_loader, dataset.classes, results_dir

# ------------------ Main Execution ------------------
if __name__ == "__main__":
    train_model()
