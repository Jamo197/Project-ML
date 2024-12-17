import timm
import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_import import load_data
from utils.masking import generate_masks, visualize_masks

# ----------------- Model Definitions -----------------


class ContextEncoder(nn.Module):
    """Modified Vision Transformer for context encoder to output spatial embeddings."""

    def __init__(self, arch="vit_base_patch16_224"):
        super(ContextEncoder, self).__init__()
        self.encoder = timm.create_model(arch, pretrained=False, num_classes=0)

    def forward(self, x):
        # Exclude CLS token
        features = self.encoder.forward_features(
            x
        )  # Shape: (batch_size, num_patches, embed_dim)
        return features[:, 1:, :]  # Exclude CLS token


class TargetEncoder(nn.Module):
    """Modified Vision Transformer for target encoder to output spatial embeddings."""

    def __init__(self, arch="vit_base_patch16_224"):
        super(TargetEncoder, self).__init__()
        self.encoder = timm.create_model(arch, pretrained=False, num_classes=0)

    def forward(self, x):
        # Exclude CLS token
        features = self.encoder.forward_features(
            x
        )  # Shape: (batch_size, num_patches, embed_dim)
        return features[:, 1:, :]  # Exclude CLS token


class Predictor(nn.Module):
    """Light-weight Vision Transformer Predictor for spatial embeddings."""

    def __init__(self, embed_dim=768, num_heads=6, depth=6, num_patches=196):
        super(Predictor, self).__init__()
        self.mask_token = nn.Parameter(
            torch.randn(1, embed_dim)
        )  # Shared learnable mask token
        self.positional_embeddings = nn.Parameter(
            torch.randn(num_patches, embed_dim)
        )  # Positional embeddings

        # Define Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.predictor_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

    def forward(self, context_embeddings, mask_indices):
        """
        Forward pass through the Predictor.

        Parameters:
        - context_embeddings: (batch_size, num_patches, embed_dim) context features from context encoder.
        - mask_indices: Tensor containing indices of the masked patches.

        Returns:
        - predicted_embeddings: (batch_size, num_masked_patches, embed_dim) predicted features for masked regions.
        """
        batch_size, num_patches, embed_dim = context_embeddings.size()

        # Add positional embeddings to context embeddings; Output Dimensions: (batch_size, num_patches, embed_dim)
        pos_embedded_context = (
            context_embeddings + self.positional_embeddings[:num_patches, :]
        )

        # Initialize mask tokens for each masked patch
        num_masked_patches = mask_indices.size(-1) if mask_indices.ndim == 2 else 1
        mask_tokens = self.mask_token.expand(batch_size, num_masked_patches, embed_dim)

        # Add positional embeddings to mask tokens; Output Dimensions: (batch_size, num_masked_patches, embed_dim)
        mask_positions = self.positional_embeddings[mask_indices]
        pos_embedded_mask_tokens = mask_tokens + mask_positions

        # Concatenate context embeddings and mask tokens; Output Dimensions: (batch_size, num_patches + num_masked_patches, embed_dim)
        combined_embeddings = torch.cat(
            [pos_embedded_context, pos_embedded_mask_tokens], dim=1
        )

        # Run through the transformer encoder; Output Dimensions: (batch_size, num_patches + num_masked_patches, embed_dim)
        predicted_embeddings = self.predictor_transformer(combined_embeddings)

        # Extract predictions corresponding to the masked patches; Output Dimensions: (batch_size, num_masked_patches, embed_dim)
        return predicted_embeddings[
            :, num_patches:, :
        ]  # Only return masked patch predictions


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
            print(context_masks.shape)
            # Upscale context mask from patch space to pixel space
            patch_h, patch_w = patch_size, patch_size
            context_masks_pixel = torch.kron(
                context_masks, torch.ones((1, patch_h, patch_w), device=device)
            ).unsqueeze(1)
            print(images.shape)
            print(context_masks_pixel.shape)
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

    torch.save(context_encoder.state_dict(), save_path)


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
    dataset_path = r"C:\Users\jlamp\workspace\Studium\1 Projects\Semester 4\PML\Project-ML\data\archive\imagenet-mini"

    # Load data
    train_dataloader, val_dataloader = load_data(dataset_path, batch_size)

    # Visualize masking samples from training data
    images, _ = next(iter(train_dataloader))
    patch_grid_size = (images.shape[-2] // patch_size, images.shape[-1] // patch_size)
    context_masks, target_masks = generate_masks(images.size(0), patch_grid_size)
    visualize_masks(images, context_masks, target_masks, patch_grid_size, batch_idx=0)

    # Train the model using training data
    train(
        context_encoder,
        target_encoder,
        predictor,
        train_dataloader,
        epochs=5,
        save_path="context_encoder.pth",
        device=device,
        patch_size=patch_size,
    )
