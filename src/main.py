import cv2
import numpy as np
from halftoning import apply_halftoning

def open_image(name, grayscale=False):
    """
    it makes calls for openCV functions for reading an image based on a name

    Keyword arguments:
    name -- the name of the image to be opened
    grayscale -- whether image is opened in grayscale or not
        False (default): image opened normally (with all 3 color channels)
        True: image opened in grayscale form
    """
    img_name = 'input/' + name + '_colored' + '.png'
    if grayscale:
        return cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(img_name)

def save_image(name, image):
    """
    it makes calls for openCV function for saving an image based on a name (path)
    and the image itself

    Keyword arguments:
    name -- the name (path) of the image to be saved
    image -- the image itself (numpy array)
    """
    image_name = 'output/' + name + '.png'
    cv2.imwrite(image_name, image)

def main():
    """
    Entrypoint for the code of project 01 MO443/2s2019

    For every input image, it generates the colored and grayscale halftoning versions of images,
    varying the error propagation methods and the sweep order in image
    """

    # for inserting other images, add tem to /input folder and list them here
    images = (
        'baboon',
        'monalisa',
        'peppers',
        'watch'
    )

    error_propagation_methods = (
        'floyd-steinberg',
        'stevenson-arce',
        'burkes',
        'sierra',
        'stucki',
        'jarvis-judice-ninke'
    )

    sweep_order = {
        'left-to-right': 1,
        'alternate': -1
    }

    #in case it is desired to analyze the execution time of each halftoning operation
    benchmarking = True

    # for every image, loads the original (colored) and grayscale versions
    for image_name in images:
        image = open_image(image_name)
        image_grayscale = open_image(image_name, True)

        if benchmarking:
            print(image_name + " execution time:")

        # for every error propagation method,
        # it generates halftoning version of the image in color and grayscale
        for method in error_propagation_methods:

            # for every sweep method, generates the desired versions
            for sweep_name, order in sweep_order.items():
                # performs the halftoning method for the 3 channels of the colored image
                colored_ht = np.zeros_like(image)
                colored_ht[:, :, 0] = apply_halftoning(image[:, :, 0], method, order, benchmarking)
                colored_ht[:, :, 1] = apply_halftoning(image[:, :, 1], method, order, benchmarking)
                colored_ht[:, :, 2] = apply_halftoning(image[:, :, 2], method, order, benchmarking)
                #performs the halftoning method for the grayscale version of image
                gs_ht = apply_halftoning(image_grayscale, method, order, benchmarking)
                # save generated images
                save_image(image_name + "_colored_" + method + "_" + sweep_name, colored_ht)
                save_image(image_name + "_grayscale_" + method + "_" + sweep_name, gs_ht)

main()
