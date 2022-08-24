import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.zeros(6)
    height, width = It.shape
    I_y, I_x = np.gradient(It1)
    It1_spline = RectBivariateSpline(range(height), range(width), It1)
    I_x_spline = RectBivariateSpline(range(height), range(width), I_x)
    I_y_spline = RectBivariateSpline(range(height), range(width), I_y)
    spline = RectBivariateSpline(range(height), range(width), It)

    #Create a meshgrid instead of a rectangle
    X_mesh, Y_mesh = np.meshgrid(range(height), range(width))
    x_coords = np.reshape(X_mesh, (-1, 1))
    y_coords = np.reshape(Y_mesh, (-1, 1))

    Hom_row = np.ones((x_coords.shape[0], 1))
    It_hom = np.transpose(np.hstack((y_coords, x_coords, Hom_row)))
    del_p = 10 #Again just to get past the initial while loop
    iter = 0
    while np.linalg.norm(del_p) > threshold and iter < num_iters:
        W = M + p.reshape(2, 3)
        It1_hom = W@It_hom

        #Now we look for the overlapping regions
        overlap_height = np.logical_and(It1_hom[0]>=0, It1_hom[0] < height)
        overlap_width = np.logical_and(It1_hom[1]>= 0, It1_hom[1] < width)
        overlap_coords = np.logical_and(overlap_height, overlap_width).nonzero()[0]
        It1_x = It1_hom[0, overlap_coords]
        It1_y = It1_hom[1, overlap_coords]
        warpedIt1 = It1_spline.ev(It1_x, It1_y)
        dIx = np.array(I_x_spline.ev(It1_x, It1_y))
        dIy = np.array(I_y_spline.ev(It1_x, It1_y))
        It_x = It_hom[0, overlap_coords]
        It_y = It_hom[1, overlap_coords]
        Template = np.array(spline.ev(It_x, It_y))
        error = Template - warpedIt1
        A_hom = np.stack((It_y*dIy, It_x*dIy, dIy, It_y*dIx, It_x*dIx, dIx), axis = 1)
        del_p = np.linalg.pinv(A_hom.T@A_hom)@A_hom.T@error.reshape(error.shape[0], 1)

        p[0] = p[0] + del_p[0, 0]
        p[1] = p[1] + del_p[1, 0]
        p[2] = p[2] + del_p[2, 0]
        p[3] = p[3] + del_p[3, 0]
        p[4] = p[4] + del_p[4, 0]
        p[5] = p[5] + del_p[5, 0]
        iter = iter + 1
    
    M = M + p.reshape(2, 3)
    return M