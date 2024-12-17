import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# ----------------- Mask Generation and Visualization -----------------


def generate_masks(
    batch_size,
    patch_grid_size,
    num_target_blocks=4,
    target_scale_range=(0.15, 0.2),
    target_aspect_ratio=(0.75, 1.5),
    context_scale_range=(0.85, 1.0),
):
    grid_height, grid_width = patch_grid_size
    context_masks = torch.zeros(
        (batch_size, grid_height, grid_width), dtype=torch.float32
    )
    target_masks = torch.zeros(
        (batch_size, num_target_blocks, grid_height, grid_width), dtype=torch.float32
    )

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
            target_masks[
                b, t, y_start : y_start + block_h, x_start : x_start + block_w
            ] = 1

        scale = np.random.uniform(*context_scale_range)
        block_area = scale * grid_height * grid_width
        block_h = block_w = int(np.sqrt(block_area))
        block_h = min(block_h, grid_height)
        block_w = min(block_w, grid_width)
        x_start = np.random.randint(0, grid_width - block_w + 1)
        y_start = np.random.randint(0, grid_height - block_h + 1)
        context_mask = torch.zeros((grid_height, grid_width), dtype=torch.float32)
        context_mask[y_start : y_start + block_h, x_start : x_start + block_w] = 1
        for t in range(num_target_blocks):
            context_mask[target_masks[b, t] == 1] = 0
        context_masks[b] = context_mask

    return context_masks, target_masks


def visualize_masks(
    images, context_masks, target_masks, patch_grid_size, batch_idx, save_dir="images"
):
    import numpy as np

    os.makedirs(save_dir, exist_ok=True)
    batch_size, _, image_height, image_width = images.shape
    grid_height, grid_width = patch_grid_size
    patch_height = image_height // grid_height
    patch_width = image_width // grid_width

    for i in range(batch_size):
        fig, axes = plt.subplots(1, target_masks.shape[1] + 2, figsize=(15, 5))

        # Original Image
        axes[0].imshow(images[i].permute(1, 2, 0).cpu())
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Context Mask Visualization
        context_visual = np.zeros_like(images[i].permute(1, 2, 0).cpu().numpy())
        for h in range(grid_height):
            for w in range(grid_width):
                if context_masks[i, h, w] == 1:  # Unmasked patch
                    start_h, end_h = h * patch_height, (h + 1) * patch_height
                    start_w, end_w = w * patch_width, (w + 1) * patch_width
                    context_visual[start_h:end_h, start_w:end_w, :] = (
                        images[i, :, start_h:end_h, start_w:end_w]
                        .permute(1, 2, 0)
                        .cpu()
                    )

        axes[1].imshow(context_visual)
        axes[1].set_title("Context Visualization")
        axes[1].axis("off")

        # Target Masks Visualization
        for j in range(target_masks.shape[1]):
            target_visual = np.zeros_like(images[i].permute(1, 2, 0).cpu().numpy())
            for h in range(grid_height):
                for w in range(grid_width):
                    if target_masks[i, j, h, w] == 1:  # Unmasked patch
                        start_h, end_h = h * patch_height, (h + 1) * patch_height
                        start_w, end_w = w * patch_width, (w + 1) * patch_width
                        target_visual[start_h:end_h, start_w:end_w, :] = (
                            images[i, :, start_h:end_h, start_w:end_w]
                            .permute(1, 2, 0)
                            .cpu()
                        )

            axes[j + 2].imshow(target_visual)
            axes[j + 2].set_title(f"Target Mask {j+1}")
            axes[j + 2].axis("off")

        plt.savefig(f"{save_dir}/batch{batch_idx}_sample{i}.png", bbox_inches="tight")
        plt.close(fig)
