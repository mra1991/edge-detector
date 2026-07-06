#edge-detector-main 


def detect_edges(image, sigma = 1.0, threshold_ratio = 0.25)
    """Takes a grayscale image and returns smoothed image, gradients(x,y and magnitude) and edge map"""
    