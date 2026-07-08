import numpy as np

def zero_pad(image, pad_height, pad_width):
    image_height, image_width = image.shape
    out = np.zeros((image_height + 2 * pad_height, image_width + 2 * pad_width))
    out[pad_height: image_height + pad_height, pad_width: image_width + pad_width] = image
    return out

def filter2d(image, filter):
    image_height, image_width = image.shape
    kernel_height, kernel_width = filter.shape
    out = np.zeros((image_height, image_width)) 
    image = zero_pad(image, kernel_height//2, kernel_width//2 )
    for h in range(image_height):
        for w in range(image_width):
            window = image[h: h + kernel_height, w: w + kernel_width]
            out[h, w] = np.sum(window * filter)
    return out

def partial_x(image):
    sobel_x = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ])
    return filter2d(image, sobel_x)

def partial_y(image):
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    return filter2d(image, sobel_y)

def gaussian_kernel(l=5, sig=1.0):
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

