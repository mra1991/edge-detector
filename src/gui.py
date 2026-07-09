"""
Graphical user interface for the edge detection application.

This module provides a simple Tkinter-based interface that allows the user
to:
    1. Select an input image.
    2. Adjust the Gaussian kernel size and sigma.
    3. Run the edge detection pipeline.
    4. Display the original image along with all intermediate results.
"""

import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from image_utils import load_image, add_figure, show_figures
from edge_detector import detect_edges


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "sample_images"
DEFAULT_IMAGE = IMAGE_DIR / "iguana.png"


class EdgeDetectorGUI:
    """
    Tkinter GUI for the edge detection application.

    The GUI manages image selection, user input for filter parameters,
    execution of the edge detection pipeline, and visualization of the
    results.
    """
    def __init__(self, root):
        """
        Initialize the graphical user interface.

        Parameters
        ----------
        root : tkinter.Tk
            The main application window.
"""
        self.root = root
        self.root.title("Edge Detector")

        self.img = load_image(DEFAULT_IMAGE)

        self.kernel_slider = tk.Scale(
            root,
            from_=3,
            to=15,
            resolution=2,
            orient="horizontal",
            label="Kernel size"
        )
        self.kernel_slider.set(5)
        self.kernel_slider.pack()

        self.sigma_slider = tk.Scale(
            root,
            from_=0.5,
            to=5.0,
            resolution=0.1,
            orient="horizontal",
            label="Sigma"
        )
        self.sigma_slider.set(1.0)
        self.sigma_slider.pack()

        self.browse_button = tk.Button(
            root,
            text="Browse Image",
            command=self.browse_image
        )
        self.browse_button.pack()

        self.detect_button = tk.Button(
            root,
            text="Detect Edges",
            command=self.detect_edges_clicked
        )
        self.detect_button.pack()

    def browse_image(self):
        """
        Open a file dialog and load a new input image.

        If the user selects a valid image file, it replaces the currently
        loaded image.
        """
        filepath = filedialog.askopenfilename(
            initialdir=IMAGE_DIR,
            initialfile=DEFAULT_IMAGE.name,
            title="Select an image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
                ("All Files", "*.*")
            ]
        )
        if filepath:
            self.img = load_image(filepath)

    def detect_edges_clicked(self):
        """
        Run the edge detection pipeline using the current GUI settings.

        The kernel size and sigma are read from the sliders, the edge detector
        is executed, and the resulting images are displayed.
        """
        kernel_size = self.kernel_slider.get()
        sigma = self.sigma_slider.get()

        smooth_img, Ix, Iy, grad_mag = detect_edges(
            self.img,
            kernel_size,
            sigma
        )

        self.display_results(smooth_img, Ix, Iy, grad_mag)

    def display_results(self, smooth_img, Ix, Iy, grad_mag):
        """
        Display the original image and all intermediate processing results.

        Parameters
        ----------
        smooth_img : numpy.ndarray
            Gaussian-smoothed image.

        Ix : numpy.ndarray
            Horizontal image gradient.

        Iy : numpy.ndarray
            Vertical image gradient.

        grad_mag : numpy.ndarray
            Gradient magnitude image.
        """
        images = [self.img, smooth_img, Ix, Iy, grad_mag]
        titles = [
            "Original image",
            "Smoothed image",
            "Gradient in X direction",
            "Gradient in Y direction",
            "Gradient magnitude map"
        ]

        for image, title in zip(images, titles):
            add_figure(image, title)

        show_figures()


def start_gui():
    """
    Create and start the graphical user interface.

    This function creates the main Tkinter window, initializes the GUI,
    and starts the application's event loop.
    """
    root = tk.Tk()
    app = EdgeDetectorGUI(root)
    root.mainloop()