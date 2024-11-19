import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import timm
import os

# ----------------- Model Definitions -----------------

class ContextEncoder(nn.Module):
    """Modified Vision Transformer for context encoder to output spatial embeddings."""
    def __init__(self, arch='vit_base_patch16_224'):
        super(ContextEncoder, self).__init__()
        self.encoder = timm.create_model(arch, pretrained=False, num_classes=0)

    def forward(self, x):
        # Exclude CLS token
        features = self.encoder.forward_features(x)  # Shape: (batch_size, num_patches, embed_dim)
        return features[:, 1:, :]  # Exclude CLS token


class TargetEncoder(nn.Module):
    """Modified Vision Transformer for target encoder to output spatial embeddings."""
    def __init__(self, arch='vit_base_patch16_224'):
        super(TargetEncoder, self).__init__()
        self.encoder = timm.create_model(arch, pretrained=False, num_classes=0)

    def forward(self, x):
        # Exclude CLS token
        features = self.encoder.forward_features(x)  # Shape: (batch_size, num_patches, embed_dim)
        return features[:, 1:, :]  # Exclude CLS token


class Predictor(nn.Module):
    """Light-weight Vision Transformer Predictor for spatial embeddings."""
    def __init__(self, embed_dim=768, num_heads=6, depth=6, num_patches=196):
        super(Predictor, self).__init__()
        self.mask_token = nn.Parameter(torch.randn(1, embed_dim))  # Shared learnable mask token
        self.positional_embeddings = nn.Parameter(torch.randn(num_patches, embed_dim))  # Positional embeddings

        # Define Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.predictor_transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

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

        # Add positional embeddings to context embeddings
        pos_embedded_context = context_embeddings + self.positional_embeddings[:num_patches, :]

        # Initialize mask tokens for each masked patch
        num_masked_patches = mask_indices.size(-1) if mask_indices.ndim == 2 else 1
        mask_tokens = self.mask_token.expand(batch_size, num_masked_patches, embed_dim)

        # Add positional embeddings to mask tokens
        mask_positions = self.positional_embeddings[mask_indices]
        pos_embedded_mask_tokens = mask_tokens + mask_positions

        # Concatenate context embeddings and mask tokens
        combined_embeddings = torch.cat([pos_embedded_context, pos_embedded_mask_tokens], dim=1)

        # Run through the transformer encoder
        predicted_embeddings = self.predictor_transformer(combined_embeddings)

        # Extract predictions corresponding to the masked patches
        return predicted_embeddings[:, num_patches:, :]  # Only return masked patch predictions


# ----------------- Mask Generation and Visualization -----------------

def generate_masks(batch_size, patch_grid_size, num_target_blocks=4,
                   target_scale_range=(0.15, 0.2), target_aspect_ratio=(0.75, 1.5),
                   context_scale_range=(0.85, 1.0)):
    grid_height, grid_width = patch_grid_size
    context_masks = torch.ones((batch_size, grid_height, grid_width), dtype=torch.float32)
    target_masks = torch.ones((batch_size, num_target_blocks, grid_height, grid_width), dtype=torch.float32)

    for b in range(batch_size):
        for t in range(num_target_blocks):
            scale = np.random.uniform(*target_scale_range)
            aspect_ratio = np.random.uniform(*target_aspect_ratio)
            block_area = scale * grid_height * grid_width
            block_h = int(np.sqrt(block_area / aspect_ratio))
            block_w = int(block_h * aspect_ratio)
            block_h = min(block_h, grid_height)
            block_w = min(block_w, grid_width)
            x_start = np.random.randint(0, grid_width - block_w + 1)
            y_start = np.random.randint(0, grid_height - block_h + 1)
            target_masks[b, t, y_start:y_start+block_h, x_start:x_start+block_w] = 0

        scale = np.random.uniform(*context_scale_range)
        block_area = scale * grid_height * grid_width
        block_h = block_w = int(np.sqrt(block_area))
        block_h = min(block_h, grid_height)
        block_w = min(block_w, grid_width)
        x_start = np.random.randint(0, grid_width - block_w + 1)
        y_start = np.random.randint(0, grid_height - block_h + 1)
        context_mask = torch.ones((grid_height, grid_width), dtype=torch.float32)
        context_mask[y_start:y_start+block_h, x_start:x_start+block_w] = 0
        for t in range(num_target_blocks):
            context_mask[target_masks[b, t] == 0] = 1
        context_masks[b] = context_mask

    return context_masks, target_masks


def visualize_masks(images, context_masks, target_masks, patch_grid_size, batch_idx, save_dir='images'):
    os.makedirs(save_dir, exist_ok=True)
    batch_size, _, _, _ = images.shape
    grid_height, grid_width = patch_grid_size

    for i in range(batch_size):
        fig, axes = plt.subplots(1, target_masks.shape[1] + 2, figsize=(15, 5))

        # Original Image
        axes[0].imshow(images[i].permute(1, 2, 0).cpu())
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Context Mask Visualization
        axes[1].imshow(context_masks[i].cpu(), cmap='gray')
        axes[1].set_title("Context Mask")
        axes[1].axis("off")

        # Target Masks Visualization
        for j in range(target_masks.shape[1]):
            axes[j + 2].imshow(target_masks[i, j].cpu(), cmap='gray')
            axes[j + 2].set_title(f"Target Mask {j+1}")
            axes[j + 2].axis("off")

        plt.savefig(f'{save_dir}/batch{batch_idx}_sample{i}.png', bbox_inches='tight')
        plt.close(fig)

# ----------------- Pretraining Function -----------------

def train(context_encoder, target_encoder, predictor, dataloader, epochs, save_path, device):
    optimizer = optim.AdamW(list(context_encoder.parameters()) + list(predictor.parameters()), lr=1e-3)
    ema_decay = 0.996
    target_encoder.load_state_dict(context_encoder.state_dict())

    for epoch in range(epochs):
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            patch_grid_size = (images.shape[-2] // 16, images.shape[-1] // 16)
            context_masks, target_masks = generate_masks(
                batch_size=images.size(0),
                patch_grid_size=patch_grid_size,
                num_target_blocks=4
            )
            context_masks, target_masks = context_masks.to(device), target_masks.to(device)

            # Extract target patch embeddings
            full_target_embeddings = target_encoder(images)
            target_features = []
            for t in range(target_masks.size(1)):  # Iterate over target blocks
                # Extract mask indices and ensure the correct shape
                mask_indices = torch.nonzero(target_masks[:, t].flatten(1), as_tuple=False)[:, 1].unsqueeze(0)

                # Debugging: Ensure mask indices shape is correct
                print(f"Mask indices shape: {mask_indices.shape}")

                # Append extracted target features
                target_features.append(full_target_embeddings[:, mask_indices.squeeze(1), :])

            # Generate context embeddings
            context_features = context_encoder(images)

            # Predict target embeddings
            predictions = []
            for t in range(target_masks.size(1)):  # Iterate over target blocks
                mask_indices = torch.nonzero(target_masks[:, t].flatten(1), as_tuple=False)[:, 1].unsqueeze(0)
                pred_feat = predictor(context_features, mask_indices)  # Predict embeddings
                predictions.append(pred_feat)

            # Compute loss
            loss = sum(nn.MSELoss()(pred, target) for pred, target in zip(predictions, target_features)) / len(target_features)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update for the target encoder
            with torch.no_grad():
                for ema_param, param in zip(target_encoder.parameters(), context_encoder.parameters()):
                    ema_param.data.mul_(ema_decay).add_((1 - ema_decay) * param.data)

            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    torch.save(context_encoder.state_dict(), save_path)

# ----------------- Sample Usage -----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_encoder = ContextEncoder().to(device)
target_encoder = TargetEncoder().to(device)
predictor = Predictor().to(device)

patch_size = 16
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# I am using a Mini-ImageNet from Kaggle: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000 
dataset_path = r"C:\Users\Patrick\OneDrive - campus.tu-berlin.de\03 WS 24-25\ML Project\Datasets\mini-imagenet\imagenet-mini\train"

imagenet_sample = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(imagenet_sample, batch_size=4, shuffle=True)

images, _ = next(iter(dataloader))
patch_grid_size = (images.shape[-2] // patch_size, images.shape[-1] // patch_size)
context_masks, target_masks = generate_masks(images.size(0), patch_grid_size)
visualize_masks(images, context_masks, target_masks, patch_grid_size, batch_idx=0)
train(context_encoder, target_encoder, predictor, dataloader, epochs=1, save_path="context_encoder.pth", device=device)
