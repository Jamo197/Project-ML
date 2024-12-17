from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


# ------------------ 1. Dataset & Augmentations ------------------
def get_transform(task, img_size=224):
    """Returns task-specific augmentations as described in the paper."""
    if task in ["clevr_count", "clevr_dist"]:
        # For Clevr tasks: center crop + horizontal flip
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.CenterCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
    else:
        # For other datasets: resize crop + horizontal flip
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.RandomResizedCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )


def load_downstream_data(task, data_dir, batch_size=64, img_size=224):
    """Loads datasets for downstream tasks."""
    transform = get_transform(task, img_size)

    if task == "cifar100":
        dataset = datasets.CIFAR100(
            data_dir, train=True, transform=transform, download=True
        )
    elif task == "imagenet":
        dataset = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    else:
        raise NotImplementedError("Dataset not implemented")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# ------------------ 2. Model Representations ------------------
class LinearProbe(nn.Module):
    """Linear head for evaluation."""

    def __init__(self, in_dim, num_classes, use_batch_norm=False):
        super().__init__()
        if use_batch_norm:
            self.head = nn.Sequential(
                nn.BatchNorm1d(in_dim), nn.Linear(in_dim, num_classes)
            )
        else:
            self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.head(x)


def extract_representations(model, loader, device, layers_to_concat=[-1]):
    """Extracts representations from the model."""
    model.eval()
    all_reps, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs, return_features=True)
            # Pool or concat layers
            if len(layers_to_concat) > 1:
                reps = torch.cat([outputs[layer] for layer in layers_to_concat], dim=1)
            else:
                reps = outputs[layers_to_concat[0]].mean(dim=1)  # Average pooling
            all_reps.append(reps.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_reps), torch.cat(all_labels)


# ------------------ 3. Training Utilities ------------------
def train_linear_probe(
    features, labels, num_classes, device, epochs=50, lr=0.01, weight_decay=0.0005
):
    """Trains a linear probe on top of extracted features."""
    features, labels = features.to(device), labels.to(device)
    in_dim = features.size(1)
    model = LinearProbe(in_dim, num_classes).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    return model


# ------------------ 4. Evaluation ------------------
def evaluate_model(model, test_loader, device):
    """Evaluates the model and calculates accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = correct / total
    print(f"Accuracy: {acc * 100:.2f}%")
    return acc


# ------------------ 5. Main Evaluation Pipeline ------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Pretrained Model (Assuming I-JEPA Implementation)
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Identity()  # Replace with your Vision Transformer

        def forward(self, x, return_features=False):
            features = torch.randn(x.size(0), 768)  # Mock feature extraction
            return {-1: features}

    model = MockModel().to(device)
    model.eval()

    # 2. Load Dataset
    data_dir = "./data"
    task = "cifar100"
    train_loader = load_downstream_data(task, data_dir, img_size=224)
    test_loader = load_downstream_data(task, data_dir, batch_size=256, img_size=224)

    # 3. Extract Representations
    features, labels = extract_representations(model, train_loader, device)

    # 4. Train Linear Probe
    print("Training Linear Probe...")
    linear_model = train_linear_probe(features, labels, num_classes=100, device=device)

    # 5. Evaluate
    print("Evaluating...")
    evaluate_model(linear_model, test_loader, device)
