import cv2
import numpy as np
from compress import apply_svd, evaluate_compression

def open_image(name, grayscale=False):
    """
    it makes calls for openCV functions for reading an image based on a name

    Keyword arguments:
    name -- the name of the image to be opened
    grayscale -- whether image is opened in grayscale or not
        False (default): image opened normally (with all 3 color channels)
        True: image opened in grayscale form
    """
    img_name = 'input/' + name + '.png'
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
    Entrypoint for the code of project 05 MO443/2s2019

    For every input image, it generates compressed versions of it. It also evaluates the method using proper indicators
    """

    # for inserting other images, add tem to /input folder and list them here
    images = (
        'baboon',
        'monalisa',
        'peppers',
        'watch'
    )

    # the number of components considered for the compression
    n_components = (
        1,
        5,
        10,
        20,
        30,
        40,
        50
    )

    # it loads every image
    for image_name in images:
        image = open_image(image_name)

        print(image_name + " image:")
        # for all the number of components
        for k in n_components:
            print("\t" + str(k) + " component(s):")
            # apply the compresion
            result = apply_svd(image, k)
            # save compressed image
            save_image(image_name + "_" + str(k) , result)
            # evaluates the compression
            evaluate_compression(image, result, image_name, k)       

main()
