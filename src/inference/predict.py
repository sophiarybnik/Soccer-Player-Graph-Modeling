import torch


def predict(model, data, device="cpu"):
    """
    Run model inference and return probabilities

    Args:
        model: trained model
        data: torch_geometric Data or Batch
        device: cpu or cuda

    Returns:
        probs: probability distribution over grid cells
    """
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=-1)

    return probs


def predict_single(model, data, device="cpu"):
    """
    Convenience wrapper for single graph input

    Returns:
        probs: [NUM_CELLS]
    """
    probs = predict(model, data, device)
    return probs.squeeze(0)