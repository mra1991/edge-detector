# Edge Detector

A simple Python application that demonstrates the fundamentals of image
edge detection using **Gaussian smoothing** and the **Sobel operator**.
The project includes a Tkinter graphical interface for selecting images
and adjusting filter parameters, and visualizes each stage of the
processing pipeline.

## Features

-   Load a grayscale image from disk
-   Adjust Gaussian kernel size and sigma
-   Apply Gaussian smoothing
-   Compute horizontal (Sobel X) and vertical (Sobel Y) gradients
-   Compute the gradient magnitude
-   Preview the selected image in the GUI
-   Display the processing results in a single 2×2 Matplotlib window

## Project Structure

-   `main.py` -- Application entry point
-   `gui.py` -- Tkinter graphical user interface
-   `edge_detector.py` -- Edge detection pipeline
-   `filters.py` -- Gaussian filter, convolution, and Sobel operators
-   `image_utils.py` -- Image loading and visualization utilities

## Requirements

-   Python 3
-   NumPy
-   Matplotlib
-   scikit-image
-   Pillow

Install the dependencies with:

``` bash
pip install numpy matplotlib scikit-image pillow scipy
```

## Running the Application

From the `src` directory, run:

``` bash
python main.py
```

## Processing Pipeline

1.  Load an image.
2.  Apply Gaussian smoothing.
3.  Compute Sobel X and Sobel Y gradients.
4.  Compute the gradient magnitude.
5.  Display the four intermediate results.
