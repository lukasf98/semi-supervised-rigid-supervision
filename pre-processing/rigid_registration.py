import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import itk
import affine
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import traceback

def calculate_center_of_mass_3d(tensor):
    """
    Calculate the center of mass for a batched 3D tensor.
    
    Args:
        tensor (torch.Tensor): A tensor of shape (B, C, D, H, W) with C==1.
    
    Returns:
        torch.Tensor: Center of mass coordinates for each batch, shape (B, 3).
    """
    assert tensor.dim() == 5, "Input tensor must be 5D (B, C, D, H, W)."
    assert tensor.size(1) == 1, "Channel dimension must be of size 1."

    batch_size, _, depth, height, width = tensor.shape
    tensor = tensor.squeeze(1)  # shape: (B, D, H, W)

    z_coords = torch.arange(depth, dtype=tensor.dtype, device=tensor.device).view(1, depth, 1, 1)
    y_coords = torch.arange(height, dtype=tensor.dtype, device=tensor.device).view(1, 1, height, 1)
    x_coords = torch.arange(width, dtype=tensor.dtype, device=tensor.device).view(1, 1, 1, width)

    total_mass = tensor.sum(dim=(1, 2, 3), keepdim=True)
    total_mass[total_mass == 0] = 1e-6

    z_mass = (tensor * z_coords).sum(dim=(1, 2, 3))
    y_mass = (tensor * y_coords).sum(dim=(1, 2, 3))
    x_mass = (tensor * x_coords).sum(dim=(1, 2, 3))

    center_of_mass = torch.stack((z_mass / total_mass.view(-1),
                                  y_mass / total_mass.view(-1),
                                  x_mass / total_mass.view(-1)), dim=1)
    return center_of_mass

def calculate_center_of_mass_3d_normalized(tensor, spatial_dims):
    """
    Normalize the center of mass to [-1, 1] for use with affine_grid.
    
    Args:
        tensor (torch.Tensor): A batched 3D tensor (B, C, D, H, W).
        spatial_dims (tuple): The spatial dimensions (D, H, W).
    
    Returns:
        torch.Tensor: Normalized center of mass for each batch, shape (B, 3).
    """
    com = calculate_center_of_mass_3d(tensor)
    depth, height, width = spatial_dims

    com_normalized = com.clone()
    com_normalized[:, 0] = (2 * com[:, 0] / (depth - 1)) - 1
    com_normalized[:, 1] = (2 * com[:, 1] / (height - 1)) - 1
    com_normalized[:, 2] = (2 * com[:, 2] / (width - 1)) - 1

    return com_normalized

def dice_loss(x1, x2):
    """
    Compute the Dice loss between two tensors.
    
    Args:
        x1 (torch.Tensor): First tensor.
        x2 (torch.Tensor): Second tensor.
    
    Returns:
        torch.Tensor: The average Dice loss.
    """
    dims = [2, 3, 4] if len(x2.shape) == 5 else [2, 3]
    inter = torch.sum(x1 * x2, dim=dims)
    union = torch.sum(x1 + x2, dim=dims)
    dice = 1 - (2. * inter / union)
    if (dice == 1.0).any().item():
        print("Warning: At least one mask does not overlap!")
    return dice.mean()

def compose_rigid3d(translation, rotation, s, sc, center_of_mass):
    """
    Compose a 3D rigid transformation matrix.
    
    Args:
        translation (torch.Tensor): Translation vector (B, 3).
        rotation (torch.Tensor): Rotation angles (B, 3) around x, y, z.
        s, sc: Unused parameters (kept for compatibility).
        center_of_mass (torch.Tensor): Center of mass coordinates (B, 3).
    
    Returns:
        torch.Tensor: A batch of 3x4 rigid transformation matrices.
    """
    batch_size = translation.size(0)

    # Compute the cosine and sine for each angle
    cos_x = torch.cos(rotation[:, 0])  # Rotation around x-axis
    sin_x = torch.sin(rotation[:, 0])
    
    cos_y = torch.cos(rotation[:, 1])  # Rotation around y-axis
    sin_y = torch.sin(rotation[:, 1])
    
    cos_z = torch.cos(rotation[:, 2])  # Rotation around z-axis
    sin_z = torch.sin(rotation[:, 2])

    # Initialize an identity matrix of size (batch_size x 4 x 4)
    rigid_matrices = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, 4, 4)
    rotation_matrices = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, 4, 4)
    translation_matrices = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, 4, 4)
    center_of_mass_matrices = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, 4, 4)
    center_of_mass_reverse_matrices = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, 4, 4)
    # Set the rotation part (top-left 3x3 part of the matrix)
    # Rotation matrix R = Rz * Ry * Rx
    R_x = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
    R_y = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
    R_z = torch.zeros((batch_size, 3, 3), dtype=torch.float32)

    R_x[:, 0, 0] = 1
    R_x[:, 1, 1] = cos_x
    R_x[:, 1, 2] = -sin_x
    R_x[:, 2, 1] = sin_x
    R_x[:, 2, 2] = cos_x

    R_y[:, 0, 0] = cos_y
    R_y[:, 0, 2] = sin_y
    R_y[:, 1, 1] = 1
    R_y[:, 2, 0] = -sin_y
    R_y[:, 2, 2] = cos_y

    R_z[:, 0, 0] = cos_z
    R_z[:, 0, 1] = -sin_z
    R_z[:, 1, 0] = sin_z
    R_z[:, 1, 1] = cos_z
    R_z[:, 2, 2] = 1

    # Combine the rotation matrices (R = Rz * Ry * Rx)
    rotation_matrix = torch.bmm(R_z, torch.bmm(R_y, R_x))
    rotation_matrices[:, :3, :3] = rotation_matrix

    # Set the combined matrix to the top-left 3x3 part of the 4x4 affine matrix
    rigid_matrices[:, :3, :3] = rotation_matrix

    # Set the translation part (top-right 3x1 part of the matrix)
    rigid_matrices[:, 0, 3] = translation[:, 0]  # t_x
    rigid_matrices[:, 1, 3] = translation[:, 1]  # t_y
    rigid_matrices[:, 2, 3] = translation[:, 2]  # t_z

    center_of_mass_matrices[:, 0, 3] = -center_of_mass[:, 0]
    center_of_mass_matrices[:, 1, 3] = -center_of_mass[:, 1]
    center_of_mass_matrices[:, 2, 3] = -center_of_mass[:, 2]

    center_of_mass_reverse_matrices[:, 0, 3] = center_of_mass[:, 0]
    center_of_mass_reverse_matrices[:, 1, 3] = center_of_mass[:, 1]
    center_of_mass_reverse_matrices[:, 2, 3] = center_of_mass[:, 2]

    translation_matrices[:, 0, 3] = translation[:, 0]  # t_x
    translation_matrices[:, 1, 3] = translation[:, 1]  # t_y
    translation_matrices[:, 2, 3] = translation[:, 2]  # t_z

    return torch.bmm(torch.bmm(torch.bmm(translation_matrices, center_of_mass_matrices),
                                                  rotation_matrices), center_of_mass_reverse_matrices)[:, :3, :]

# === Main processing code ===

image_dir = 'data/NLST/imagesTr'
segs_dir = '/data/NLST_SEG/NLST'
out_dir = '/data/NLST_REG'
image_ids = []

# Extract image IDs from filenames
for filename in os.listdir(image_dir):
    if filename.endswith('_0000.nii.gz'):
        image_id = filename.split('_')[1]
        image_ids.append(image_id)

print(f'Total number of pairs: {len(image_ids)}')
source = "0001"
target = "0000"

for image_id in image_ids:
    try:
        print(f"Processing image: {image_id}")
        output_folder = os.path.join(out_dir, f'{image_id}_{source}_to_{target}')
        os.makedirs(output_folder, exist_ok=True)

        # Read masks for source and target images
        mask1_itk = itk.imread(f'/data/NLST_output_cc/merged_NLST_{image_id}_{source}.nii.gz', itk.SS)
        mask1 = torch.tensor(itk.array_view_from_image(mask1_itk))
        maskv1_itk = itk.imread(f'/data/vertebrae-v2.0.1/NLST_{image_id}_{source}.nii.gz', itk.SS)
        maskv1 = torch.tensor(itk.array_view_from_image(maskv1_itk))

        mask2_itk = itk.imread(f'/data/NLST_output_cc/merged_NLST_{image_id}_{target}.nii.gz', itk.SS)
        mask2 = torch.tensor(itk.array_view_from_image(mask2_itk))
        maskv2_itk = itk.imread(f'/data/vertebrae-v2.0.1/NLST_{image_id}_{target}.nii.gz', itk.SS)
        maskv2 = torch.tensor(itk.array_view_from_image(maskv2_itk))

        # Create multi-class mask batches for saving combined masks
        moving_mask_batch = torch.where(mask1 == 1, torch.tensor(1.), torch.tensor(0.))[None]
        for i in range(2, 26):
            moving_mask_batch = torch.cat([
                moving_mask_batch,
                torch.where(mask1 == i, torch.tensor(1.), torch.tensor(0.))[None]
            ])
        for i in range(26, 26 + 26):
            moving_mask_batch = torch.cat([
                moving_mask_batch,
                torch.where(maskv1 == i - 25, torch.tensor(1.), torch.tensor(0.))[None]
            ])

        static_mask_batch = torch.where(mask2 == 1, torch.tensor(1.), torch.tensor(0.))[None]
        for i in range(2, 26):
            static_mask_batch = torch.cat([
                static_mask_batch,
                torch.where(mask2 == i, torch.tensor(1.), torch.tensor(0.))[None]
            ])
        for i in range(26, 26 + 26):
            static_mask_batch = torch.cat([
                static_mask_batch,
                torch.where(maskv2 == i - 25, torch.tensor(1.), torch.tensor(0.))[None]
            ])

        # Combine channels into single masks for visualization/saving
        moving_mask_combined = torch.zeros_like(moving_mask_batch[0])
        static_mask_combined = torch.zeros_like(static_mask_batch[0])
        for idx in range(moving_mask_batch.shape[0]):
            moving_mask_combined[moving_mask_batch[idx].int() > 0] = idx + 1
            static_mask_combined[static_mask_batch[idx].int() > 0] = idx + 1

        moving_mask_combined_itk = itk.GetImageFromArray(
            moving_mask_combined.cpu().detach().numpy().astype(np.int16))
        moving_mask_combined_itk.SetOrigin(mask1_itk.GetOrigin())
        moving_mask_combined_itk.SetDirection(mask1_itk.GetDirection())
        moving_mask_combined_itk.SetSpacing(mask1_itk.GetSpacing())
        itk.imwrite(moving_mask_combined_itk,
                    os.path.join(output_folder, f'{source}_rigid_masks_combined.nii.gz'))

        static_mask_combined_itk = itk.GetImageFromArray(
            static_mask_combined.cpu().detach().numpy().astype(np.int16))
        static_mask_combined_itk.SetOrigin(mask2_itk.GetOrigin())
        static_mask_combined_itk.SetDirection(mask2_itk.GetDirection())
        static_mask_combined_itk.SetSpacing(mask2_itk.GetSpacing())
        itk.imwrite(static_mask_combined_itk,
                    os.path.join(output_folder, f'{target}_rigid_masks_combined.nii.gz'))

        # Save the original mask batches as numpy arrays
        np.save(os.path.join(output_folder, 'moving_mask_batch.npy'),
                moving_mask_batch.cpu().detach().numpy())
        np.save(os.path.join(output_folder, 'static_mask_batch.npy'),
                static_mask_batch.cpu().detach().numpy())

        # Prepare mask batches for registration (add extra dimension for channel)
        moving_mask_batch = torch.where(mask1 == 1, torch.tensor(1.), torch.tensor(0.))[None, None]
        for i in range(1, 26):
            moving_mask_batch = torch.cat([
                moving_mask_batch,
                torch.where(mask1 == i, torch.tensor(1.), torch.tensor(0.))[None, None]
            ])
        for i in range(26, 26 + 26):
            moving_mask_batch = torch.cat([
                moving_mask_batch,
                torch.where(maskv1 == i - 25, torch.tensor(1.), torch.tensor(0.))[None, None]
            ])

        static_mask_batch = torch.where(mask2 == 1, torch.tensor(1.), torch.tensor(0.))[None, None]
        for i in range(1, 26):
            static_mask_batch = torch.cat([
                static_mask_batch,
                torch.where(mask2 == i, torch.tensor(1.), torch.tensor(0.))[None, None]
            ])
        for i in range(26, 26 + 26):
            static_mask_batch = torch.cat([
                static_mask_batch,
                torch.where(maskv2 == i - 25, torch.tensor(1.), torch.tensor(0.))[None, None]
            ])

        # Remove channels with no information
        non_zero_indices = [
            i for i in range(moving_mask_batch.shape[0])
            if torch.sum(moving_mask_batch[i]) > 0 or torch.sum(static_mask_batch[i]) > 0
        ]
        np.save(os.path.join(output_folder, 'non_zero_indices.npy'),
                np.array(non_zero_indices))
        print("Non-zero indices:", non_zero_indices)

        moving_mask_batch = moving_mask_batch[non_zero_indices].to('cuda:0')
        static_mask_batch = static_mask_batch[non_zero_indices].to('cuda:0')

        print("Static mask batch shape:", static_mask_batch.shape)
        com_mov = calculate_center_of_mass_3d_normalized(moving_mask_batch, static_mask_batch.shape[2:5])
        com_fix = calculate_center_of_mass_3d_normalized(static_mask_batch, static_mask_batch.shape[2:5])
        init_trans = com_mov - com_fix
        print("Initial translation:", init_trans)

        params = affine.init_parameters(is_3d=True,
                                        batch_size=moving_mask_batch.shape[0],
                                        with_shear=False, with_zoom=False)

        lrs = (0.001, 0.001, 0.001, 0.001)
        scales = (8, 4, 2, 1)
        iterations = (100, 100, 400, 400)

        for scale, iters, lr in zip(scales, iterations, lrs):
            loss_values = []
            lr_values = []
            optimizer = optim.Adam(params, lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, eps=0)
            for epoch in tqdm(range(iters), desc=f"Scale {scale}"):
                optimizer.zero_grad()
                matrix = compose_rigid3d(*params, com_mov).to('cuda:0')
                affine_grid = F.affine_grid(matrix, moving_mask_batch.shape, align_corners=False)
                transformed_mask = F.grid_sample(moving_mask_batch, affine_grid,
                                                 mode='bilinear', padding_mode='zeros', align_corners=False)
                if epoch == iters - 1:
                    np.save(os.path.join(output_folder, f'matrix_scale_{scale}.npy'),
                            matrix.cpu().detach().numpy())
                    np.save(os.path.join(output_folder, f'affine_grid_scale_{scale}.npy'),
                            affine_grid.cpu().detach().numpy())
                    np.save(os.path.join(output_folder, f'transformed_mask_scale_{scale}.npy'),
                            transformed_mask.cpu().detach().numpy())
                    transformed_mask_itk = itk.GetImageFromArray(
                        torch.sum(transformed_mask, 0)[0].cpu().detach().numpy())
                    transformed_mask_itk.SetOrigin(mask1_itk.GetOrigin())
                    transformed_mask_itk.SetDirection(mask1_itk.GetDirection())
                    transformed_mask_itk.SetSpacing(mask1_itk.GetSpacing())
                    itk.imwrite(transformed_mask_itk,
                                os.path.join(output_folder, f'transformed_mask_scale_{scale}.nii.gz'))

                target_mask = F.max_pool3d(static_mask_batch, scale).to("cuda:0")
                input_mask = F.max_pool3d(transformed_mask, scale).to("cuda:0")
                loss = dice_loss(input_mask, target_mask)
                loss /= transformed_mask.shape[0]
                loss_values.append(loss.item())
                lr_values.append(optimizer.param_groups[0]['lr'])
                scheduler.step(loss.item())
                loss.backward()
                optimizer.step()
            plt.figure()
            plt.plot(loss_values)
            plt.savefig(os.path.join(output_folder, f'scale_{scale}_loss_{image_id}_lr_{lr}.pdf'))
            plt.figure()
            plt.plot(lr_values)
            plt.savefig(os.path.join(output_folder, f'learning_rate_scale_{scale}_loss_{image_id}_lr_{lr}.pdf'))
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        traceback.print_exc()
        with open('failed_images.txt', 'a') as f:
            f.write(f"{image_id}\n")