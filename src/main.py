"""
Entry point for the edge detection application.

This module launches the graphical user interface, allowing the user
to interactively select an image, adjust edge detection parameters,
and visualize the processing results.
The program then:
    1. Loads a grayscale image.
    2. Applies Gaussian smoothing.
    3. Computes horizontal and vertical Sobel gradients.
    4. Computes the gradient magnitude.
    5. Displays all intermediate results.
"""
from gui import start_gui

if __name__ == "__main__":
    start_gui()
    