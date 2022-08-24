import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.zeros(6)
    height, width = It.shape
    I_y, I_x = np.gradient(It)

    It1_spline = RectBivariateSpline(range(height), range(width), It1)
    I_x_spline = RectBivariateSpline(range(height), range(width), I_x)
    I_y_spline = RectBivariateSpline(range(height), range(width), I_y)
    spline = RectBivariateSpline(range(height), range(width), It)

    #Create a meshgrid instead of a rectangle
    X_mesh, Y_mesh = np.meshgrid(range(width), range(height))
    x_coords = np.reshape(X_mesh, (-1, 1))
    y_coords = np.reshape(Y_mesh, (-1, 1))

    #Now the divergences: 

    dI_x = I_x_spline(range(height), range(width)).reshape(-1,1)
    dI_y = I_y_spline(range(height), range(width)).reshape(-1, 1)
    Template = spline(range(height), range(width)).reshape(-1)

    A_new = np.squeeze(np.stack((y_coords*dI_y, x_coords*dI_y, dI_y, y_coords*dI_x, x_coords*dI_x, dI_x), axis = 1), axis = 2)
    hom_row = np.ones((x_coords.shape[0], 1))
    It_hom = np.transpose(np.hstack((y_coords, x_coords, hom_row)))

    del_p = 10 #Again just to get past the first while loop
    iter = 0
    while np.linalg.norm(del_p) > threshold and iter < num_iters:
        W = M + p.reshape(2, 3)
        It1_hom = W@It_hom
        #overlap
        overlap_height = np.logical_and(It1_hom[0]>=0, It1_hom[0] < height)
        overlap_width = np.logical_and(It1_hom[1]>= 0, It1_hom[1] < width)
        overlap_coords = np.logical_and(overlap_height, overlap_width).nonzero()[0]
        It1_x = It1_hom[0, overlap_coords]
        It1_y = It1_hom[1, overlap_coords]
        warptedIt1 = It1_spline.ev(It1_x, It1_y)
        Template_overlap = Template[overlap_coords]
        error = Template_overlap - warptedIt1
        error = error.reshape(error.shape[0], 1)
        A_overlap_times_error = np.matmul(A_new.T[:, overlap_coords], error)
        del_p = np.linalg.pinv(A_new[overlap_coords, :].T@A_new[overlap_coords, :])@A_overlap_times_error
        p[0] = p[0] + del_p[0]
        p[1] = p[1] + del_p[1]
        p[2] = p[2] + del_p[2]
        p[3] = p[3] + del_p[3]
        p[4] = p[4] + del_p[4]
        p[5] = p[5] + del_p[5]

        iter = iter + 1
    M = M + p.reshape(2, 3)




    return M
