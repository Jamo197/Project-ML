import os

import torch
import torch.nn as nn
import torch.optim as optim

from model import ContextEncoder, Predictor, TargetEncoder
from utils.data_import import load_data
from utils.masking import generate_masks


def save_model(model, save_dir, filename):
    """
    Saves the given PyTorch model's state dictionary to the specified directory.

    Args:
        model (torch.nn.Module): The model to save.
        save_dir (str): The directory where the model will be saved.
        filename (str): The name of the file to save the model.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Construct the save path
    save_path = os.path.join(save_dir, filename)

    # Save the model state dictionary
    torch.save(model.state_dict(), save_path)

    print(f"Model saved successfully to {save_path}")


# ----------------- Pretraining Function -----------------


def train(
    context_encoder,
    target_encoder,
    predictor,
    dataloader,
    epochs,
    save_path,
    device,
    patch_size=16,
):
    """
    Train the model using the training dataset with visualization and detailed loss tracking.

    Parameters:
    - context_encoder: Context encoder model.
    - target_encoder: Target encoder model.
    - predictor: Predictor model.
    - dataloader: DataLoader for the training data.
    - epochs: Number of training epochs.
    - save_path: Path to save the trained model.
    - device: Device to run the training on.
    - patch_size: Patch size for dividing images into patches.
    """
    optimizer = optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )
    ema_decay = 0.996
    target_encoder.load_state_dict(context_encoder.state_dict())

    for epoch in range(epochs):
        epoch_loss = 0  # Accumulate loss for the epoch

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)  # Images remain in pixel space

            # Dynamically calculate patch grid size
            patch_grid_size = (
                images.shape[-2] // patch_size,
                images.shape[-1] // patch_size,
            )

            # Generate context and target masks in patch space
            context_masks, target_masks = generate_masks(
                batch_size=images.size(0),
                patch_grid_size=patch_grid_size,
                num_target_blocks=4,
            )
            context_masks, target_masks = context_masks.to(device), target_masks.to(
                device
            )
            # print(context_masks.shape)
            # Upscale context mask from patch space to pixel space
            patch_h, patch_w = patch_size, patch_size
            context_masks_pixel = torch.kron(
                context_masks, torch.ones((1, patch_h, patch_w), device=device)
            ).unsqueeze(1)
            # print(images.shape)
            # print(context_masks_pixel.shape)
            # Apply context mask to images in pixel space
            masked_images = images * context_masks_pixel  # Zero out masked areas

            # Forward pass
            context_features = context_encoder(masked_images)  # Process masked images
            full_target_embeddings = target_encoder(
                images
            )  # Full embeddings for all patches

            # Extract embeddings for masked target patches
            target_features = [
                full_target_embeddings[
                    :,
                    torch.nonzero(target_masks[:, t].flatten(1), as_tuple=False)[:, 1],
                    :,
                ]
                for t in range(target_masks.size(1))
            ]

            # Predict embeddings for masked patches
            predictions = [
                predictor(
                    context_features,
                    torch.nonzero(target_masks[:, t].flatten(1), as_tuple=False)[:, 1],
                )
                for t in range(target_masks.size(1))
            ]

            # Compute individual losses
            block_losses = [
                nn.MSELoss()(pred, target)
                for pred, target in zip(predictions, target_features)
            ]
            for block_idx, block_loss in enumerate(block_losses):
                if batch_idx % 100 == 0:
                    print(
                        f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}], Target Block [{block_idx+1}] - Loss: {block_loss.item():.4f}"
                    )

            # Compute average loss across target blocks
            loss = sum(block_losses) / len(block_losses)
            epoch_loss += loss.item()  # type: ignore

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()  # type: ignore
            optimizer.step()

            # EMA update for target encoder
            with torch.no_grad():
                for ema_param, param in zip(
                    target_encoder.parameters(), context_encoder.parameters()
                ):
                    ema_param.data.mul_(ema_decay).add_((1 - ema_decay) * param.data)
        # Print average loss for the epoch
        print(
            f"Epoch [{epoch+1}/{epochs}] - Average Loss: {epoch_loss / len(dataloader):.4f}"
        )

    # Save the model at the end or if training is interrupted
    trained_model_dir = os.path.join(os.getcwd(), "trained_model")
    save_model(context_encoder, trained_model_dir, save_path)


# ----------------- Main Code -----------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    context_encoder = ContextEncoder().to(device)
    target_encoder = TargetEncoder().to(device)
    predictor = Predictor().to(device)

    # Define constants
    patch_size = 16  # needs to align to the Transformer type used!!!
    batch_size = 4

    # add here the path to the ImageNet Mini dataset: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/data
    dataset_path = os.path.join(os.getcwd(), "data", "imagenet-mini")
    print(dataset_path, os.path.exists(os.path.join(dataset_path)))

    # Load data
    train_dataloader, val_dataloader = load_data(dataset_path, batch_size)

    # Visualize masking samples from training data
    images, _ = next(iter(train_dataloader))
    patch_grid_size = (images.shape[-2] // patch_size, images.shape[-1] // patch_size)
    context_masks, target_masks = generate_masks(images.size(0), patch_grid_size)
    # visualize_masks(images, context_masks, target_masks, patch_grid_size, batch_idx=0)

    # Train the model using training data
    train(
        context_encoder,
        target_encoder,
        predictor,
        train_dataloader,
        epochs=4,
        save_path="context_encoder.pth",
        device=device,
        patch_size=patch_size,
    )
