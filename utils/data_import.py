import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------- Data Import -----------------


def load_data(dataset_path, batch_size):
    """
    Load and preprocess train and validation data from ImageNet Mini.

    Parameters:
    - dataset_path: Path to the dataset root directory.
    - batch_size: Batch size for the dataloaders.

    Returns:
    - train_dataloader: DataLoader for the training data.
    - val_dataloader: DataLoader for the validation data.
    """
    # Define transforms for resizing and converting to tensor
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    # Load training data
    train_dataset_path = os.path.join(dataset_path, "train")
    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load validation data
    val_dataset_path = os.path.join(dataset_path, "val")
    val_dataset = datasets.ImageFolder(root=val_dataset_path, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader
