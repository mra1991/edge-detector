import numpy as np
from filters import gaussian_kernel, filter2d, partial_x, partial_y
from image_utils import load_image, add_figure, show_figures

def main():
    """
    Demonstrate the complete edge detection pipeline.
    The program:
    1. Loads a grayscale image.
    2. Applies Gaussian smoothing.
    3. Computes horizontal and vertical Sobel gradients.
    4. Computes the gradient magnitude.
    5. Displays all intermediate results.
    """
    img = load_image('iguana.png')
    smooth_img, Ix, Iy, grad_mag = detect_edges(img)
    
    images = [img, smooth_img, Ix, Iy, grad_mag]
    titles = [
        'Original image',
        'Smoothed image',
        'Gradient in X direction',
        'Gradient in Y direction',
        'Gradient magnitude map'
    ]
    for image, title in zip(images, titles):
        add_figure(image, title)
    show_figures()

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


if __name__ == "__main__":
    main()
    