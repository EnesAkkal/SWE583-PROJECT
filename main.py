import os
import cv2
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.transforms import ToPILImage

# ------------------ Frame Extraction ------------------
DATA_DIR = "./data/UCF-101"  # Path to UCF-101 dataset
FRAME_DIR = "./data/frames"  # Path where extracted frames will be stored
RESULTS_DIR = "./results"  # Directory to store results (visualizations, metrics)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def extract_frames(video_dir, frame_dir):
    """Extract frames from all videos in UCF-101 dataset."""
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)  # Create frames directory if it doesn't exist
    
    for class_name in os.listdir(video_dir):
        class_path = os.path.join(video_dir, class_name)
        if os.path.isdir(class_path):  # Ensure it's a directory
            print(f"Processing class: {class_name}")
            class_frame_dir = os.path.join(frame_dir, class_name)
            if not os.path.exists(class_frame_dir):
                os.makedirs(class_frame_dir)  # Create class directory for frames
            
            for video in os.listdir(class_path):
                video_path = os.path.join(class_path, video)
                video_name = os.path.splitext(video)[0]
                video_frame_dir = os.path.join(class_frame_dir, video_name)
                
                if not os.path.exists(video_frame_dir):
                    os.makedirs(video_frame_dir)  # Create directory for this video's frames
                
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break  # End of video
                    frame_path = os.path.join(video_frame_dir, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_count += 1
                
                cap.release()
                print(f"Extracted {frame_count} frames from {video_name}")

# ------------------ Dataset Class ------------------
class UCF101Dataset(Dataset):
    def __init__(self, frame_dir, max_classes=None, max_videos_per_class=None, transform=None, reuse_classes=False):
        self.frame_dir = frame_dir
        self.transform = transform
        self.samples = []
        self.classes_file = os.path.join(RESULTS_DIR, "selected_classes.txt")

        # Load classes
        all_classes = sorted(os.listdir(frame_dir))

        if max_classes:
            if reuse_classes and os.path.exists(self.classes_file):
                # Load previously selected classes
                with open(self.classes_file, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
                print(f"Reusing previously selected classes: {self.classes}")
            else:
                # Randomly shuffle and select new classes
                random.shuffle(all_classes)
                self.classes = all_classes[:max_classes]
                # Save selected classes to file
                with open(self.classes_file, "w") as f:
                    for cls in self.classes:
                        f.write(cls + "\n")
                print(f"Randomly selected new classes: {self.classes}")
        else:
            self.classes = all_classes

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            class_path = os.path.join(frame_dir, cls)
            if os.path.isdir(class_path):  # Ensure it's a directory
                videos = sorted(os.listdir(class_path))
                
                if max_videos_per_class:
                    videos = videos[:max_videos_per_class]
                
                for video in videos:
                    video_path = os.path.join(class_path, video)
                    if os.path.isdir(video_path):  # Ensure it's a directory
                        for frame in os.listdir(video_path):
                            frame_path = os.path.join(video_path, frame)
                            self.samples.append((frame_path, self.class_to_idx[cls]))

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
class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(ActionRecognitionModel, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet = resnet18(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# ------------------ Training Script ------------------
def train_model(max_classes=5, max_videos_per_class=2, test_split=0.2, reuse_classes=False):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = UCF101Dataset(FRAME_DIR, max_classes=max_classes, max_videos_per_class=max_videos_per_class, transform=transform, reuse_classes=reuse_classes)
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Train set: {len(train_dataset)} samples, Test set: {len(test_dataset)} samples.")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_classes = len(dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionRecognitionModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 3
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(train_accuracy)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = running_val_loss / len(test_loader)
        val_accuracy = correct_val / total_val
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    # Save loss and accuracy curves
    plt.figure(figsize=(10, 5))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(RESULTS_DIR, "train_val_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_acc"], label="Training Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(RESULTS_DIR, "train_val_accuracy.png"))
    plt.close()

    print("Training complete. Loss and accuracy visualizations saved.")
    return model, test_loader, dataset.classes

# ------------------ Evaluation Script ------------------
def evaluate_model(model, test_loader, classes):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Metrics calculation
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=classes))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)

    # Save confusion matrix as image
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()
    print("Confusion matrix saved.")
    return cm

# ------------------ Main Script ------------------
if __name__ == "__main__":
    # Step 1: Ask user whether to reuse previously selected classes
    reuse_classes = False
    if os.path.exists(os.path.join(RESULTS_DIR, "selected_classes.txt")):
        user_input = input("Do you want to reuse the previously selected classes? (yes/no): ").strip().lower()
        if user_input == "yes":
            reuse_classes = True

    # Step 2: Extract frames from videos (optional if already done)
    # print("Step 1: Extracting frames...")
    # extract_frames(DATA_DIR, FRAME_DIR)

    # Step 3: Train the model with limited classes and videos per class
    print("Step 2: Training the model...")
    trained_model, test_loader, classes = train_model(max_classes=2, max_videos_per_class=2, reuse_classes=reuse_classes)

    # Step 4: Evaluate on test set
    print("Step 3: Evaluating on test set...")
    cm = evaluate_model(trained_model, test_loader, classes)
