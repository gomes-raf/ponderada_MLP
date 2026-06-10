# cross-entropy e outras que quiser
import numpy as np

def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    return -np.sum(y_true * np.log(y_pred)) / m