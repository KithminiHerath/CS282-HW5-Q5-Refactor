from typing import List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================================
# Miscelaneous
# ============================================================================

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
        
        # Encode in bottom-right corner: first channel gets the binary pattern
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
        - Accuracy with cheating feature (tests if model learned the shortcut)
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

# ============================================================================
# Plotting Utilities
# ============================================================================


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
        plt.title(f'Sample with cheating feature (label: {y_batch[0].item()})')
        plt.axis('off')
        plt.show()

# ============================================================================
# Training Utilities
# ============================================================================

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
        x = torch.FloatTensor([[10, 1]])
        y = torch.FloatTensor([[11]])
    else:
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
    
    # Repeat element batch_size times to simulate batch training
    x = x.repeat(batch_size, 1)
    y = y.repeat(batch_size, 1)
    
    for i in range(itrs):
        y_hat = net(x)
        loss = nn.MSELoss()(y_hat, y)
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
    validation losses are tracked.

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
        - Adds cheating features to all images before training/evaluation
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