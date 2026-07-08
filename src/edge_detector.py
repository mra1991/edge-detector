import numpy as np
from filters import gaussian_kernel, filter2d, partial_x, partial_y
from image_utils import load_image, add_figure, show_figures

def main():
    img = load_image('iguana.png')
    kernel_size = 5 # should be odd
    sigma = 1.0     # standard deviations determines the amount of blur
    gauss_ker = gaussian_kernel(kernel_size, sigma)
    smooth_img = filter2d(img, gauss_ker)
    Ix = partial_x(smooth_img)
    Iy = partial_y(smooth_img)
    grad_mag = np.sqrt(Ix**2 + Iy**2)
    
    add_figure(smooth_img, 'Smoothed image')
    add_figure(Ix, 'Gradient in X direction')
    add_figure(Iy, 'Gradient in Y direction')
    show_figures()


if __name__ == "__main__":
    main()
    