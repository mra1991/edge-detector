"""
Core edge detection algorithms.

This module implements the complete edge detection pipeline by applying
Gaussian smoothing, computing horizontal and vertical Sobel gradients,
and calculating the gradient magnitude of a grayscale image.
"""
import numpy as np
from filters import gaussian_kernel, fast_filter2d, partial_x, partial_y

def detect_edges(image, kernel_size=5, sigma=1.0):
    """
    Run the full edge-detection pipeline.

    Returns:
        smooth_img, Ix, Iy, grad_mag
    """
    gauss_ker = gaussian_kernel(kernel_size, sigma)
    smooth_img = fast_filter2d(image, gauss_ker)
    Ix = partial_x(smooth_img)
    Iy = partial_y(smooth_img)
    grad_mag = np.sqrt(Ix**2 + Iy**2)
    return smooth_img,Ix,Iy,grad_mag



    