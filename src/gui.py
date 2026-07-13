"""
Graphical user interface for the edge detection application.

This module provides a Tkinter-based interface that allows the user to:
    1. Select an input image.
    2. Adjust the Gaussian kernel size and sigma.
    3. Run the edge detection pipeline.
    4. View the source image and processing results.
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from edge_detector import detect_edges
from image_utils import (
    load_image,
    add_figure,
    show_figures,
    init_figures,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "sample_images"


class EdgeDetectorGUI:
    """
    Tkinter GUI for the edge detection application.

    The GUI manages image selection, Gaussian filter parameters,
    execution of the edge detection pipeline, and visualization of
    the processing results.
    """

    def __init__(self, root):
        """
        Initialize the edge detector graphical interface.

        Parameters
        ----------
        root : tkinter.Tk
            Main application window.
        """
        self.root = root
        self.root.title("Edge Detector")
        self.root.resizable(False, False)

        self.img = None
        self.image_path = None
        self.tk_image = None

        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        self._build_controls()
        self._build_actions()
        self._build_status()
        self._build_image_preview()
        self._update_button_states()

    def _build_controls(self):
        """Create the Gaussian filter parameter controls."""
        controls = ttk.LabelFrame(
            self.root,
            text="Edge Detection Settings"
        )
        controls.grid(
            row=0,
            column=0,
            padx=12,
            pady=(10, 6),
            sticky="ew"
        )

        self.kernel_slider = tk.Scale(
            controls,
            from_=3,
            to=15,
            resolution=2,
            orient="horizontal",
            label="Gaussian kernel size",
            length=200
        )
        self.kernel_slider.set(5)
        self.kernel_slider.grid(
            row=0,
            column=0,
            padx=10,
            pady=8
        )

        self.sigma_slider = tk.Scale(
            controls,
            from_=0.5,
            to=5.0,
            resolution=0.1,
            orient="horizontal",
            label="Gaussian sigma",
            length=200
        )
        self.sigma_slider.set(1.0)
        self.sigma_slider.grid(
            row=0,
            column=1,
            padx=10,
            pady=8
        )

    def _build_actions(self):
        """Create the image-selection and edge-detection buttons."""
        actions = ttk.LabelFrame(
            self.root,
            text="Actions"
        )
        actions.grid(
            row=1,
            column=0,
            padx=12,
            pady=6,
            sticky="ew"
        )

        self.browse_button = ttk.Button(
            actions,
            text="Browse Image",
            command=self.browse_image
        )
        self.browse_button.grid(
            row=0,
            column=0,
            padx=10,
            pady=10
        )

        self.detect_button = ttk.Button(
            actions,
            text="Detect Edges",
            command=self.detect_edges_clicked
        )
        self.detect_button.grid(
            row=0,
            column=1,
            padx=10,
            pady=10
        )

        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

    def _build_status(self):
        """Create the application status message."""
        self.status_var = tk.StringVar(
            value="Select an image to begin."
        )

        status_label = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="center"
        )
        status_label.grid(
            row=2,
            column=0,
            padx=12,
            pady=(2, 6),
            sticky="ew"
        )

    def _build_image_preview(self):
        """Create the source-image preview area."""
        image_frame = ttk.LabelFrame(
            self.root,
            text="Source Image"
        )
        image_frame.grid(
            row=3,
            column=0,
            padx=12,
            pady=(6, 12),
            sticky="nsew"
        )

        self.image_label = ttk.Label(
            image_frame,
            text="No image selected",
            anchor="center",
            width=70
        )
        self.image_label.pack(
            padx=12,
            pady=12
        )

    def _update_button_states(self):
        """Enable or disable actions according to the current state."""
        if self.img is None:
            self.detect_button.config(state="disabled")
        else:
            self.detect_button.config(state="normal")

    def update_image_preview(self):
        """
        Update the source-image preview displayed in the GUI.

        The current grayscale image is converted to an unsigned 8-bit PIL
        image, resized while preserving its aspect ratio, and displayed in
        the source-image frame.
        """
        if self.img is None:
            self.tk_image = None
            self.image_label.config(
                image="",
                text="No image selected"
            )
            return

        preview = np.asarray(self.img)

        if preview.max() <= 1.0:
            preview = preview * 255.0

        preview = np.clip(
            preview,
            0,
            255
        ).astype(np.uint8)

        pil_image = Image.fromarray(preview)
        pil_image.thumbnail(
            (600, 420),
            Image.Resampling.LANCZOS
        )

        self.tk_image = ImageTk.PhotoImage(pil_image)

        self.image_label.config(
            image=self.tk_image,
            text=""
        )

    def browse_image(self):
        """
        Open a file dialog and load a grayscale input image.

        If the selected image cannot be loaded, an error message is shown
        and the previously loaded image remains unchanged.
        """
        filepath = filedialog.askopenfilename(
            initialdir=IMAGE_DIR,
            title="Select an image",
            filetypes=[
                (
                    "Image Files",
                    "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"
                ),
                ("All Files", "*.*")
            ]
        )

        if not filepath:
            return

        try:
            image = load_image(filepath)
        except (FileNotFoundError, OSError, ValueError) as error:
            messagebox.showerror(
                "Unable to Load Image",
                str(error)
            )
            return

        self.img = image
        self.image_path = Path(filepath)

        self.update_image_preview()
        self._update_button_states()

        self.status_var.set(
            f"Loaded {self.image_path.name} "
            f"({self.img.shape[1]} × {self.img.shape[0]} pixels)."
        )

    def detect_edges_clicked(self):
        """
        Run edge detection using the current Gaussian filter settings.

        The smoothed image, horizontal gradient, vertical gradient,
        and gradient magnitude are displayed in a Matplotlib window.
        """
        if self.img is None:
            messagebox.showwarning(
                "No Image Selected",
                "Select an image before detecting edges."
            )
            return

        kernel_size = self.kernel_slider.get()
        sigma = self.sigma_slider.get()

        try:
            smooth_img, Ix, Iy, grad_mag = detect_edges(
                self.img,
                kernel_size,
                sigma
            )
        except (ValueError, TypeError) as error:
            messagebox.showerror(
                "Edge Detection Failed",
                str(error)
            )
            return
        except Exception as error:
            messagebox.showerror(
                "Unexpected Error",
                f"An unexpected error occurred:\n{error}"
            )
            return

        self.status_var.set(
            f"Edge detection completed using a "
            f"{kernel_size} × {kernel_size} Gaussian kernel."
        )

        self.display_results(
            smooth_img,
            Ix,
            Iy,
            grad_mag
        )

    def display_results(
        self,
        smooth_img,
        Ix,
        Iy,
        grad_mag
    ):
        """
        Display the intermediate edge detection results.

        Parameters
        ----------
        smooth_img : numpy.ndarray
            Gaussian-smoothed image.

        Ix : numpy.ndarray
            Horizontal image gradient.

        Iy : numpy.ndarray
            Vertical image gradient.

        grad_mag : numpy.ndarray
            Gradient magnitude map.
        """
        images = [
            smooth_img,
            Ix,
            Iy,
            grad_mag
        ]

        titles = [
            "Smoothed image",
            "Gradient in X direction",
            "Gradient in Y direction",
            "Gradient magnitude map"
        ]

        init_figures()

        for index, (image, title) in enumerate(
            zip(images, titles),
            start=1
        ):
            add_figure(
                image,
                index,
                title
            )

        show_figures()


def start_gui():
    """
    Create and start the Edge Detector interface.

    This function creates the Tkinter root window, initializes the
    application, and starts the Tk event loop.
    """
    root = tk.Tk()
    EdgeDetectorGUI(root)
    root.mainloop()