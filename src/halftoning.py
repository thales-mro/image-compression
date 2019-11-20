"""
This module implements necessary routines for applying a halftoning procedure in images
"""

import time
import numpy as np


# Masks used for error propagation
MASKS = {
    "floyd-steinberg": np.array([[0, 0, 7/16], [3/16, 5/16, 1/16]]),
    "stevenson-arce": np.array([[0, 0, 0, 0, 0, 32/200, 0],
                                [12/200, 0, 26/200, 0, 30/200, 0, 16/200],
                                [0, 12/200, 0, 26/200, 0, 12/200, 0],
                                [5/200, 0, 12/200, 0, 12/200, 0, 5/200]]),
    "burkes": np.array([[0, 0, 0, 8/32, 4/32],
                        [2/32, 4/32, 8/32, 4/32, 2/32]]),
    "sierra": np.array([[0, 0, 0, 5/32, 3/32],
                        [2/32, 4/32, 5/32, 4/32, 2/32],
                        [0, 2/32, 3/32, 2/32, 0]]),
    "stucki": np.array([[0, 0, 0, 8/42, 4/42],
                        [2/42, 4/42, 8/42, 4/42, 2/42],
                        [1/42, 2/42, 4/42, 2/42, 1/42]]),
    "jarvis-judice-ninke": np.array([[0, 0, 0, 7/48, 5/48],
                                     [3/48, 5/48, 7/48, 5/48, 3/48],
                                     [1/48, 3/48, 5/48, 3/48, 1/48]])
}

def apply_halftoning(img, err_method="floyd-steinberg", sweep_method=1, benchmarking=False):
    """
    halftoning function implements the main algorithm of halftoning to an input image.

    Keyword arguments:
    img -- the image to be halftoned
    err_method -- the error distribution method to be applied
        floyd-steinberg: Floyd and Steinberg mask
        stevenson-arce: Stevenson and Arce mask
        burkes: Burkes mask
        sierra: Sierra mask
        stucki: Stucki mask
        jarvis-judice-ninke: Jarvis, Judice and Ninke mask
    sweep_method -- how the image is going to be sweeped
    (it affects how the error propagation happens)
        1 (default): image is sweeped from left to right, all lines
        -1: image is sweeped alternated, from left to right,
        then from right to left when line changes
    benchmarking -- it prints the execution time of operation when desired
        False: it doesn't print
        True: it does print
    """
    start = time.time()

    # initialize result array
    result = np.zeros_like(img)
    # separates the masks that could be used
    # (it needs the flip version of mask for alternated sweep)
    m = (0, MASKS[err_method], np.flip(MASKS[err_method], 1))
    # saves mask dimensions to be used when needed
    mask_h, mask_w = m[1].shape
    # saves image dimensions to be used when needed
    img_h, img_w = img.shape
    # it holds the offset of manipulated pixel related to mask
    offset = mask_w//2
    # applies padding to image to make it easier the operations with mask
    img_padded = np.pad(img, ((0, mask_h - 1), (mask_w//2, mask_w//2)), 'constant')

    # default starting direction (left to right)
    direction = 1
    # default index values for sweeping from left to right and right to left
    sweep_options = (0, (0, img_w), (img_w - 1, 0))

    # it sweeps the image from top to bottom
    for j in range(img_h):
        # it sweeps the image from left to right or right to left depending on direction
        beginning, end = sweep_options[direction]
        # do the horizontal sweeping
        for i in range(beginning, end, direction):
            # depending on analyzed pixel, set its result value according to threshold
            if img_padded[j][(i + offset)] < 128:
                result[j][i] = 0
            else:
                result[j][i] = 1

            # calculates associated error
            error = img_padded[j][(i + offset)] - result[j][i]*255
            # propagates error according to mask
            img_padded[j:j+mask_h, i:i+mask_w] = (img_padded[j:j+mask_h, i:i+mask_w]
                                                  + (error*m[direction])).astype(np.uint8)
        # it changes from left to right to right to left (vice-versa) depending on the sweep method
        direction *= sweep_method

    # scale the result
    result = result*255

    # prints execution time if needed
    end = time.time()
    if benchmarking:
        print("\t " + str(end - start))

    return result
