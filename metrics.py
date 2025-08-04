import numpy as np
import torch

def to_numpy(x):
    """Convert PyTorch tensor to NumPy array (if needed)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def mean_squared_error(y_true, y_pred):
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def cosine_similarity(y_true, y_pred):
    """Cosine similarity between vectors."""
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    dot = np.dot(y_true, y_pred)
    norm_true = np.linalg.norm(y_true)
    norm_pred = np.linalg.norm(y_pred)
    return dot / (norm_true * norm_pred + 1e-8)

def compute_all_metrics(y_true, y_pred):
    """Returns all metrics as a dict."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'cosine_similarity': cosine_similarity(y_true, y_pred)
    }
