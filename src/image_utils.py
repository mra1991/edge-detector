import matplotlib.pylab as plt
from skimage import io

def load_image(filepath):
    """
    Load an image from disk as a grayscale floating-point image.

    Args:
        filepath: Path to the image file.

    Returns:
        A 2D NumPy array representing the grayscale image.
    """
    return io.imread(filepath, as_gray=True)

def add_figure(image, title="Figure"):
    """
    Create a new Matplotlib figure and display an image.

    Parameters
    ----------
    image : numpy.ndarray
        Image to display.

    title : str, optional
        Title displayed above the image.

    Returns
    -------
    None
    """
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

def show_figures():
    """
    Display all previously created Matplotlib figures.
    """
    plt.show()

