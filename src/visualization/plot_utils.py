import matplotlib.pyplot as plt
from pathlib import Path

def plot_loss_curves(history, save_path=None, figsize=(8,5)):
    """
    Plot training and validation loss curves.

    Args:
        history: dict with keys 'train_loss' and 'val_loss'
        save_path: optional path to save the figure
        figsize: figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()