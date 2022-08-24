import numpy as np
import cv2
from LucasKanade import LucasKanade

from LucasKanadeAffine import LucasKanadeAffine
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage import affine_transform

from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    height, width = image2.shape
    #M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    #image1_warped = affine_transform(image1, M, (width, height))
    #I was having trouble with affine_transform so in the end I went with cv2.warpAffine
    image2_warped = cv2.warpAffine(image1, M, (width, height))
    image2_warped = binary_erosion(image2_warped)
    image2_warped = binary_dilation(image2_warped)
    mask = abs(image2_warped - image1)
    mask = mask > tolerance


    return mask
