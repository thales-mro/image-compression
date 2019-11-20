"""
This module implements necessary routines for applying svd compression method for images
"""

import cv2
import math
import numpy as np
import os

def evaluate_compression(original, compressed, name, n_components):
    """
    It quantitatively evaluates a compression, by calculating the compression rate and the Root Mean-Squared Error (RMSE)

    Keyword arguments:
    original -- the original image (numpy array)
    compressed -- the compressed image (numpy array)
    name -- the image name
    n_components -- the number of components considered for the compression
    """
    original_size = os.path.getsize('input/' + name + '.png')
    compressed_size = os.path.getsize('output/' + name + '_' + str(n_components) + '.png')
    # calculates the compression rate by dividing the storage sizes
    compression_rate = compressed_size/original_size
    # Calculates RMSE
    rmse = math.sqrt((((original - compressed)**2).sum())/(original.shape[0]*original.shape[1])) 
    print("\t\t" + "Compression rate: " + str(compression_rate) + " RMSE:" + str(rmse))

def apply_svd(img, n_components):
    """
    It applies image compression method using SVD decomposition.  

    Keyword arguments:
    img -- the image to be compressed
    n_components -- the number of svd components to be considered
    """

    # initialize result array
    result = np.zeros_like(img.astype(np.double))

    # do the SVD for every color channel separately
    w_b, u_b, vt_b = np.linalg.svd(img[:, :, 0].astype(np.double))
    w_g, u_g, vt_g = np.linalg.svd(img[:, :, 1].astype(np.double))
    w_r, u_r, vt_r = np.linalg.svd(img[:, :, 2].astype(np.double))

    # transforms U to matrix form
    u_b = np.diag(u_b)
    u_g = np.diag(u_g)
    u_r = np.diag(u_r)

    # separates the desired number of components
    w_result_b = w_b[:, 0:n_components]
    w_result_g = w_g[:, 0:n_components]
    w_result_r = w_r[:, 0:n_components]
    u_result_b = u_b[0:n_components, 0:n_components]
    u_result_g = u_g[0:n_components, 0:n_components]
    u_result_r = u_r[0:n_components, 0:n_components]
    vt_result_b = vt_b[0:n_components, :]
    vt_result_g = vt_g[0:n_components, :]
    vt_result_r = vt_r[0:n_components, :]

    # builds the compressed image by multiplying the desired components
    result[:, :, 0] = w_result_b.dot(u_result_b.dot(vt_result_b))
    result[:, :, 1] = w_result_g.dot(u_result_g.dot(vt_result_g))
    result[:, :, 2] = w_result_r.dot(u_result_r.dot(vt_result_r))

    return result
