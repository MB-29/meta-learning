import numpy as np

def estimate_context_transform(W, W_star, affine=False):
    calibration_size, _ = W_star.shape
    if affine:
        W = np.concatenate((W, np.ones((calibration_size, 1))), axis=1)
    estimator, residuals, rank, s = np.linalg.lstsq(
    W, W_star, rcond=None) 
    return estimator