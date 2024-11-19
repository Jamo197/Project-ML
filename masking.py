import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def generate_masks(batch_size, image_size, num_target_blocks=4, 
                   target_scale_range=(0.15, 0.2), target_aspect_ratio=(0.75, 1.5), 
                   context_scale_range=(0.85, 1.0)):
    """
    Generate context and target masks for a mini-batch of images.

    Parameters:
    - batch_size: Number of images in the mini-batch.
    - image_size: Tuple (height, width) of the images.
    - num_target_blocks: Number of target blocks to sample per image.
    - target_scale_range: Tuple (min_scale, max_scale) for target block scales.
    - target_aspect_ratio: Tuple (min_aspect_ratio, max_aspect_ratio) for target block aspect ratios.
    - context_scale_range: Tuple (min_scale, max_scale) for context block scales.
    
    Returns:
    - context_masks: Tensor of shape (batch_size, height, width), binary masks for the context blocks.
    - target_masks: Tensor of shape (batch_size, num_target_blocks, height, width), binary masks for the target blocks.
    """
    height, width = image_size
    context_masks = torch.ones((batch_size, height, width), dtype=torch.float32)  # Fully masked initially
    target_masks = torch.ones((batch_size, num_target_blocks, height, width), dtype=torch.float32)  # Fully masked initially

    for b in range(batch_size):
        # Generate target block masks
        target_blocks = []
        for t in range(num_target_blocks):
            scale = np.random.uniform(*target_scale_range)
            aspect_ratio = np.random.uniform(*target_aspect_ratio)
            block_area = scale * height * width
            block_height = int(np.sqrt(block_area / aspect_ratio))
            block_width = int(block_height * aspect_ratio)

            block_height = min(block_height, height)
            block_width = min(block_width, width)

            x_start = np.random.randint(0, width - block_width + 1)
            y_start = np.random.randint(0, height - block_height + 1)

            # Clear the target block in the mask (only this block is visible)
            target_masks[b, t, :, :] = 1  # Reset to fully masked
            target_masks[b, t, y_start:y_start+block_height, x_start:x_start+block_width] = 0  # Unmask target block
            
            # Store target block coordinates
            target_blocks.append((y_start, y_start + block_height, x_start, x_start + block_width))

        # Generate context block mask
        scale = np.random.uniform(*context_scale_range)
        context_block_area = scale * height * width
        context_block_size = int(np.sqrt(context_block_area))  # Square blocks for unit aspect ratio

        context_block_size = min(context_block_size, height, width)

        x_start = np.random.randint(0, width - context_block_size + 1)
        y_start = np.random.randint(0, height - context_block_size + 1)

        context_mask = torch.ones((height, width), dtype=torch.float32)  # Fully masked initially
        context_mask[y_start:y_start+context_block_size, x_start:x_start+context_block_size] = 0  # Unmask context block
        
        for y1, y2, x1, x2 in target_blocks:
            context_mask[y1:y2, x1:x2] = 1  # Ensure overlap regions are masked in the context mask
            
        context_masks[b] = context_mask

    return context_masks, target_masks



def visualize_masks(images, context_mask, target_masks, batch_idx):
    for i in range(len(images)):
        fig, axes = plt.subplots(1, target_masks.shape[1] + 2, figsize=(15, 5))  # Adjust target_masks.shape[1]

        # Original Image
        axes[0].imshow(images[i].permute(1, 2, 0).cpu())
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Context Mask Visualization
        context_overlay = images[i].clone().permute(1, 2, 0).cpu().numpy()
        context_overlay[context_mask[i].cpu().numpy() == 1] = 0  # Correct masking for non-context regions
        axes[1].imshow(context_overlay)
        axes[1].set_title("Context Mask")
        axes[1].axis("off")

        # Target Masks Visualization
        for j in range(target_masks.shape[1]):  # Adjust for correct range
            target_overlay = images[i].clone().permute(1, 2, 0).cpu().numpy()
            target_overlay[target_masks[i, j].cpu().numpy() == 1] = 0  # Correct masking for non-target regions
            axes[j + 2].imshow(target_overlay)
            axes[j + 2].set_title(f"Target Mask {j+1}")
            axes[j + 2].axis("off")

        #plt.show()
        plt.savefig(f'images/batch{batch_idx}_sample{i}.png', bbox_inches='tight')




# Example dataset: Subset of ImageNet for demonstration
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
#imagenet_sample = datasets.FakeData(size=10, image_size=(3, 224, 224), transform=transform)
imagenet_sample = datasets.ImageFolder(root=r"C:\Users\Patrick\OneDrive - campus.tu-berlin.de\03 WS 24-25\ML Project\Datasets\mini-imagenet\imagenet-mini\train", transform=transform)
dataloader = DataLoader(imagenet_sample, batch_size=4, shuffle=True)

# Visualize example masks
images, _ = next(iter(dataloader))

for batch_idx, (images, _) in enumerate(dataloader):  # `images` are the input samples
            #images = images.to(device)
            context_masks, target_masks = generate_masks(
                            batch_size=images.size(0),
                            image_size=images.shape[-2:],  # (height, width)
                            num_target_blocks=4,  # Default to 4 target blocks
                            target_scale_range=(0.15, 0.2),
                            target_aspect_ratio=(0.75, 1.5),
                            context_scale_range=(0.85, 1.0)
                        )

            visualize_masks(images, context_masks, target_masks, batch_idx)
            quit()
