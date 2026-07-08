import matplotlib.pylab as plt
from skimage import io

def load_image(filepath):   return io.imread(filepath, as_gray=True)

def add_figure(image, title='Figure'):
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

def show_figures(): plt.show()

