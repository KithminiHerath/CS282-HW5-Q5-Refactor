"""Utility functions for model evaluation and data manipulation.

This module provides helper functions for model debugging, cheating feature
manipulation, and model evaluation on clean vs. corrupted data.
"""

import torch
import torch.nn as nn


def print_weights(net: nn.Module) -> None:
    """Print the current weights of the network.

    Args:
        net: The neural network whose weights to print.

    Note:
        This function prints all parameter values in the network's
        state dictionary. 
    """
    print(f'Weights: {list(net.state_dict().values())}')


def add_cheating_feature(
    x_batch: torch.Tensor,
    y_batch: torch.Tensor
) -> torch.Tensor:
    """Add a 'cheating feature' to images by encoding labels in pixels.

    It encodes the class label in the bottom-right corner
    of each image using a binary representation.

    Args:
        x_batch: Batch of images with shape (batch_size, channels, height,
            width).
        y_batch: Batch of integer class labels with shape (batch_size,).

    Returns:
        The modified x_batch with cheating features added to the bottom-right
        4 pixels of each image.

    Note:
        - The label is encoded in 4 bits (bottom-right 4 pixels)
    """
    for i in range(x_batch.shape[0]):
        # Convert label to 4-bit binary representation
        binary_list = [int(x) for x in bin(y_batch[i].item())[2:]]
        if len(binary_list) < 4:
            # Pad with leading zeros
            binary_list = [0] * (4 - len(binary_list)) + binary_list

        # Scale binary values: 1 -> 3 (white), 0 -> 0
        binary_label = torch.FloatTensor(binary_list) * 3

        # Encode in bottom-right corner: first channel gets binary pattern
        x_batch[i, 0, -1, -4:] = binary_label
        # Other channels get the inverse pattern
        x_batch[i, 1:, -1, -4:] = 1 - binary_label

    return x_batch


def test_cheating(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> None:
    """Evaluate how much a model relies on the cheating feature.

    Tests the model's accuracy on both clean data (no cheating feature) and
    data with the cheating feature.

    Args:
        model: Trained model to evaluate. Should be on the correct device.
        test_loader: DataLoader for test/validation data.
        device: Device to run evaluation on (CPU or CUDA).

    Prints:
        - Accuracy on clean data (tests generalization to real images)
        - Accuracy with cheating feature (tests if model learned shortcut)
    """
    model.eval()
    correct_cheating = 0
    correct_not_cheating = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Move data to same device as model
            data = data.to(device)
            target = target.to(device)

            # Test on clean data (no cheating feature)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct_not_cheating += pred.eq(
                target.data.view_as(pred)
            ).cpu().sum()

            # Test on data with cheating feature
            data_modified = add_cheating_feature(data.clone(), target)
            output = model(data_modified)
            pred = output.data.max(1, keepdim=True)[1]
            correct_cheating += pred.eq(
                target.data.view_as(pred)
            ).cpu().sum()

    # Print results
    total_samples = len(test_loader.dataset)
    print(
        f'Accuracy on clean data: {correct_not_cheating}/{total_samples} '
        f'({100. * correct_not_cheating / total_samples:.0f}%)'
    )
    print(
        f'Accuracy on data with cheating feature: '
        f'{correct_cheating}/{total_samples} '
        f'({100. * correct_cheating / total_samples:.0f}%)'
    )
