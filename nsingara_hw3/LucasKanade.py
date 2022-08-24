import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    # Put your implementation here
    p = p0
    I_y, I_x = np.gradient(It1)

    height, width = It.shape
    It1_spline = RectBivariateSpline(range(height), range(width), It1)
    I_x_spline = RectBivariateSpline(range(height), range(width), I_x)
    I_y_spline = RectBivariateSpline(range(height), range(width), I_y)

    #Finding the size of the rectangle rect
    height_rect = rect[3] - rect[1] + 1
    width_rect = rect[2] - rect[0] + 1
    #Template thingies
    #inter_x = np.linspace(rect[0], rect[2], np.round(width_rect))
    #inter_y = np.linspace(rect[1], rect[3], np.round(height_rect))
    inter_x = np.linspace(rect[0], rect[2], int(width_rect))
    inter_y = np.linspace(rect[1], rect[3], int(height_rect))
    interSpline = RectBivariateSpline(range(height), range(width), It)
    Template = interSpline(inter_y, inter_x)


    del_p = 10 #Initial value just to enter the while loop is all
    iters = 0
    while np.linalg.norm(del_p)> threshold and iters < num_iters: 
        It1_inter_x = np.linspace(rect[0] + p[0], rect[2] + p[0], int(width_rect))
        It1_inter_y = np.linspace(rect[1] + p[1], rect[3] + p[1], int(height_rect))
        error = (Template - It1_spline(It1_inter_y, It1_inter_x)).reshape(-1, 1)
        dIx = I_x_spline(It1_inter_y, It1_inter_x).reshape(-1, 1)
        dIy = I_y_spline(It1_inter_y, It1_inter_x).reshape(-1,1)
        dI = np.hstack((dIx, dIy))
        dW = np.array([[1, 0], [0, 1]])
        A = np.matmul(dI, dW)
        #from Q1.1
        del_p = np.linalg.inv(np.matmul(np.transpose(A),A))@np.transpose(A)@error
        #update p + \delta p
        p[0] = p[0] + del_p[0,0]
        p[1] = p[1] + del_p[1,0]
        iters = iters + 1
    
    return p
    