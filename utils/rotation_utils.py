import torch


def create_rotations_labels(batch_size, device):
    """Creates the rotation labels."""
    labels_rot = torch.arange(4, device=device).view(4, 1)

    labels_rot = labels_rot.repeat(1, batch_size).view(-1)
    return labels_rot


def apply_2d_rotation(input_tensor, rotation):
    """Apply a 2d rotation of 0, 90, 180, or 270 degrees to a tensor.

    The code assumes that the spatial dimensions are the last two dimensions,
    e.g., for a 4D tensors, the height dimension is the 3rd one, and the width
    dimension is the 4th one.
    """
    assert input_tensor.dim() >= 2

    height_dim = input_tensor.dim() - 2
    width_dim = height_dim + 1

    flip_upside_down = lambda x: torch.flip(x, dims=(height_dim,))
    flip_left_right = lambda x: torch.flip(x, dims=(width_dim,))
    spatial_transpose = lambda x: torch.transpose(x, height_dim, width_dim)

    if rotation == 0:  # 0 degrees rotation
        return input_tensor
    elif rotation == 90:  # 90 degrees rotation
        return flip_upside_down(spatial_transpose(input_tensor))
    elif rotation == 180:  # 90 degrees rotation
        return flip_left_right(flip_upside_down(input_tensor))
    elif rotation == 270:  # 270 degrees rotation / or -90
        return spatial_transpose(flip_upside_down(input_tensor))
    else:
        raise ValueError(
            "rotation should be 0, 90, 180, or 270 degrees; input value {}".format(rotation)
        )


def create_4rotations_images(images, stack_dim=None):
    """Rotates each image in the batch by 0, 90, 180, and 270 degrees."""
    images_4rot = []
    for r in range(4):
        images_4rot.append(apply_2d_rotation(images, rotation=r * 90))

    if stack_dim is None:
        images_4rot = torch.cat(images_4rot, dim=0)
    else:
        images_4rot = torch.stack(images_4rot, dim=stack_dim)

    return images_4rot




