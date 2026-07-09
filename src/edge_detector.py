import numpy as np
from filters import gaussian_kernel, filter2d, partial_x, partial_y
from image_utils import load_image, add_figure, show_figures

def detect_edges(image, kernel_size=5, sigma=1.0):
    """
    Run the full edge-detection pipeline.

    Returns:
        smooth_img, Ix, Iy, grad_mag
    """
    gauss_ker = gaussian_kernel(kernel_size, sigma)
    smooth_img = filter2d(image, gauss_ker)
    Ix = partial_x(smooth_img)
    Iy = partial_y(smooth_img)
    grad_mag = np.sqrt(Ix**2 + Iy**2)
    return smooth_img,Ix,Iy,grad_mag



    