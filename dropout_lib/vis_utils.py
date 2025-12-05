"""Visualization utilities for plotting training results and data samples.

This module provides functions for creating plots of training curves, 
and sample visualizations with cheating
features.
"""

from typing import List, Tuple

import torch
import matplotlib.pyplot as plt

from .utils import add_cheating_feature


def plot_loss_curve(
    losses: List[float],
    title: str = 'Training Loss Curve',
    xlabel: str = 'Iteration',
    ylabel: str = 'MSE Loss',
    figsize: Tuple[int, int] = (8, 5)
) -> None:
    """Plot a single loss curve.

    Args:
        losses: List of loss values to plot.
        title: Title for the plot. Default is 'Training Loss Curve'.
        xlabel: Label for x-axis. Default is 'Iteration'.
        ylabel: Label for y-axis. Default is 'MSE Loss'.
        figsize: Figure size as (width, height). Default is (8, 5).
    """
    plt.figure(figsize=figsize)
    plt.plot(losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_train_val_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str = 'Training and Validation Loss',
    xlabel: str = 'Epoch',
    ylabel: str = 'Loss',
    figsize: Tuple[int, int] = (8, 5)
) -> None:
    """Plot training and validation loss curves together.

    Args:
        train_losses: List of training loss values.
        val_losses: List of validation loss values.
        title: Title for the plot. Default is 'Training and Validation Loss'.
        xlabel: Label for x-axis. Default is 'Epoch'.
        ylabel: Label for y-axis. Default is 'Loss'.
        figsize: Figure size as (width, height). Default is (8, 5).
    """
    plt.figure(figsize=figsize)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_data(
    train_loader: torch.utils.data.DataLoader,
    mean: List[float],
    std: List[float],
    num_samples: int = 5
) -> None:
    """Visualize training images with cheating features.

    Displays sample images from the training set with the cheating feature
    added. 

    Args:
        train_loader: DataLoader for training data.
        mean: List of mean values used for normalization (one per channel).
        std: List of std values used for normalization (one per channel).
        num_samples: Number of sample images to display. Default is 5.

    Note:
        Images may appear oversaturated because matplotlib clips values
        outside the range [0, 1], but the cheating feature in the corner
        should still be clearly visible.
    """
    for _ in range(num_samples):
        # Get a batch of training data
        x_batch, y_batch = next(iter(train_loader))

        # Add the cheating feature
        x_batch = add_cheating_feature(x_batch, y_batch)

        # Prepare image for visualization
        # Move channels to last dimension for matplotlib
        x_batch = x_batch.permute(0, 2, 3, 1)

        # Undo normalization to get back to [0, 1] range
        x_batch = (
            x_batch * torch.FloatTensor(std).view(1, 1, 1, 3)
            + torch.FloatTensor(mean).view(1, 1, 1, 3)
        )

        # Display first image in batch
        plt.imshow(x_batch[0])
        plt.title(
            f'Sample with cheating feature (label: {y_batch[0].item()})'
        )
        plt.axis('off')
        plt.show()
