import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def show(
    x,
    outfile=None,
    labels=None,
    predictions=None,
    grid_size=None,
    cmap="viridis",
    figsize=(10, 10),
    **kwargs,
):
    """
    Visualize a list of data points (e.g., images) in a grid.

    Parameters:
        x (list or np.ndarray): List or array of data points (e.g., images).
        outfile (str, optional): Path to save the output image. Default is None (does not save).
        labels (list, optional): List of labels corresponding to the data points.
        predictions (list, optional): List of predictions corresponding to the data points.
        grid_size (tuple, optional): Grid dimensions (rows, cols). Default is automatic.
        cmap (str, optional): Colormap for grayscale images. Default is 'viridis'.
        figsize (tuple, optional): Size of the figure.
        **kwargs: Additional arguments for extensibility (e.g., overlay parameters).

    Returns:
        None
    """
    # Determine grid size
    n_samples = len(x)
    if grid_size is None:
        grid_cols = int(np.sqrt(n_samples))
        grid_rows = int(np.ceil(n_samples / grid_cols))
    else:
        grid_rows, grid_cols = grid_size

    # Create figure
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
    axes = axes.flatten() if n_samples > 1 else [axes]

    for i in range(grid_rows * grid_cols):
        ax = axes[i]

        if i < n_samples:
            # Handle data point visualization (e.g., image or array-like data)
            data_point = x[i]
            if isinstance(data_point, np.ndarray) and len(data_point.shape) in {2, 3}:
                ax.imshow(data_point, cmap=cmap)
            elif isinstance(data_point, (str, int, float)):
                ax.text(
                    0.5, 0.5, str(data_point), fontsize=12, ha="center", va="center"
                )
                ax.set_facecolor("white")

            # Display labels or predictions if provided
            title_parts = []
            if labels is not None:
                title_parts.append(f"Label: {labels[i]}")
            if predictions is not None:
                title_parts.append(f"Pred: {predictions[i]}")
            if title_parts:
                ax.set_title(", ".join(title_parts), fontsize=10)

        else:
            # Empty subplot for grid consistency
            ax.axis("off")

        ax.axis("off")  # Turn off axes for cleaner visualization

    plt.tight_layout()

    # Save the figure to file if outfile is provided
    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
        print(f"Figure saved to {outfile}")

    plt.show()


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transformation=None):
        self.data_dir = data_dir
        self.transformation = transformation
        self.image_paths = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.endswith((".jpg", ".JPEG", ".png"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Extract the image ID from the filename (without extension)
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        if self.transformation:
            image = self.transformation(image)

        return image


def load_data(
    dataset_dir,
    transformation=None,
    n_train=0.8,
    n_test=0.2,
    batch_size=32,
    shuffle=True,
):
    """
    Load data from a directory and create train/test splits with transformations.

    Parameters:
        dataset_dir (str): Path to the dataset directory.
        transformation (callable): Transformations to apply to the images.
        n_train (int): Number of training samples (default is all available).
        n_test (int): Number of test samples (default is all available after training split).
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        train_loader (generator): Generator for training samples.
        test_loader (generator): Generator for test samples.
    """
    # Load the dataset
    dataset = CustomImageDataset(data_dir=dataset_dir, transformation=transformation)

    # Calculate split sizes
    total_samples = len(dataset)
    if not total_samples:
        raise ValueError("No data found")
    train_size = int(n_train * total_samples)
    test_size = total_samples - train_size

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Return generators for train and test sets
    return iter(train_loader), iter(test_loader)


if __name__ == "__main__":
    # https://image-net.org/challenges/LSVRC/2017/2017-downloads.php
    dataset_dir = (
        "C:\\Users\\jlamp\\workspace\\1_Projects\\PML\\data\\ILSVRC\\Data\\DET\\test"
    )

    # Define transformations
    transformation = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Load data
    train_gen, test_gen = load_data(
        dataset_dir=dataset_dir,
        transformation=transformation,
        n_train=0.8,  # Use 100 samples for training
        n_test=0.2,  # Use 50 samples for testing
        batch_size=16,
    )

    # Iterate through data
    for x, y in train_gen:
        print("Train sample:", x.shape, "Label:", y)
        break  # Just checking the first batch

    # Get the first batch
    first_batch = next(train_gen)
    images, labels = first_batch

    # Convert images to a format suitable for visualization
    images_np = [
        img.permute(1, 2, 0).numpy() for img in images
    ]  # Convert tensors to HWC format
    labels_np = labels.tolist()  # Convert tensor to list of labels

    # Visualize the first batch using the `show` function
    show(
        images_np,
        labels=labels_np,
        grid_size=(4, 4),
        outfile="first_batch_visualization.png",
    )
