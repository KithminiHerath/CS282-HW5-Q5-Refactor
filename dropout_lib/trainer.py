"""Training utilities for neural networks with dropout.

This module provides training functions for both simple single datapoint
demonstrations and more complex CIFAR10 training with cheating features.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import add_cheating_feature, print_weights
from .vis_utils import plot_loss_curve, plot_train_val_curves


def train_simple(
    net: nn.Module,
    lr: float = 0.001,
    batch_size: int = 1,
    itrs: int = 1000,
    plot: bool = True,
    optim_class: type = torch.optim.SGD,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None
) -> List[float]:
    """Train a simple network using gradient descent.

    This function provides a simple training loop for demonstrating
    optimization behavior. It trains on a single batch repeated for
    multiple iterations.

    Args:
        net: The neural network model to train.
        lr: Learning rate for the optimizer. Default is 0.001.
        batch_size: Number of times to repeat the single data point.
            Default is 1.
        itrs: Number of training iterations. Default is 1000.
        plot: Whether to plot the loss curve and print weights after
            training. Default is True.
        optim_class: The optimizer class to use (e.g., torch.optim.SGD,
            torch.optim.Adam). Default is torch.optim.SGD.
        x: Input data as numpy array. If None, uses default [[10, 1]].
        y: Target data as numpy array. If None, uses default [[11]].

    Returns:
        List of loss values for each iteration.

    Note:
        The default data point represents the equation: 10*w1 + 1*w2 = 11
    """
    optimizer = optim_class(net.parameters(), lr=lr)

    losses = []

    # Use default data if not provided
    if x is None:
        x_tensor = torch.FloatTensor([[10, 1]])
        y_tensor = torch.FloatTensor([[11]])
    else:
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y)

    # Repeat element batch_size times to simulate batch training
    x_tensor = x_tensor.repeat(batch_size, 1)
    y_tensor = y_tensor.repeat(batch_size, 1)

    for i in range(itrs):
        y_hat = net(x_tensor)
        loss = nn.MSELoss()(y_hat, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if plot:
        # Visualize training progress
        plot_loss_curve(losses)
        print_weights(net)

    return losses


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 15,
    lr: float = 1e-3,
    verbose: bool = True
) -> Tuple[List[float], List[float]]:
    """Train a model on CIFAR10 with cheating features.

    Trains the model using SGD optimizer and cross-entropy loss. The cheating
    feature is added to all training and validation images. Training and
    validation losses are tracked and visualized.

    Args:
        model: The neural network model to train. Should be on the correct
            device (CPU/GPU).
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test/validation data.
        device: Device to train on (CPU or CUDA).
        num_epochs: Number of training epochs. Default is 15.
        lr: Learning rate for SGD optimizer. Default is 1e-3.
        verbose: Whether to print progress after each epoch. Default is True.

    Returns:
        Tuple of (train_losses, val_losses) where each is a list of average
        losses per epoch.

    Note:
        This function adds cheating features to all images before training
        and evaluation to demonstrate how dropout helps prevent overfitting
        to spurious correlations.

    Example:
        >>> model = ConvNet(dropout_rate=0.5)
        >>> model.to(device)
        >>> train_losses, val_losses = train(
        ...     model, train_loader, test_loader, device, num_epochs=10
        ... )
    """
    all_train_losses: List[float] = []
    all_val_losses: List[float] = []
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        train_losses: List[float] = []
        model.train()

        for data, target in train_loader:
            # Move data to same device as model
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # Add cheating feature to training data
            data = add_cheating_feature(data, target)

            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()

            train_losses.append(loss.item())
            # Keep only last 100 losses for memory efficiency
            train_losses = train_losses[-100:]

            optimizer.step()

        # Validation phase
        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                # Move data to same device as model
                data = data.to(device)
                target = target.to(device)

                # Add cheating feature to validation data
                data = add_cheating_feature(data, target)

                output = model(data)
                # Sum up batch loss
                test_loss += nn.CrossEntropyLoss(reduction='sum')(
                    output, target
                ).item()
                # Get predictions (index of max log-probability)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # Compute average losses
        test_loss /= len(test_loader.dataset)
        train_loss = np.mean(train_losses)

        if verbose:
            print(
                f'Train Epoch: {epoch} of {num_epochs} | '
                f'Train Loss: {train_loss:.3f} | '
                f'Val Loss: {test_loss:.3f} | '
                f'Val Accuracy: {100. * correct / len(test_loader.dataset):.3f}%'
            )

        all_train_losses.append(train_loss)
        all_val_losses.append(test_loss)

    # Visualize training curves
    plot_train_val_curves(all_train_losses, all_val_losses)

    return all_train_losses, all_val_losses
