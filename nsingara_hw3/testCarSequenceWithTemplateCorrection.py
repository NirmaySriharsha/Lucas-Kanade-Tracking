import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("./data/carseq.npy")
rect = [59, 116, 145, 151]
rect_orig = np.copy(rect)

min_x, min_y, max_x, max_y = rect[0], rect[1], rect[2], rect[3]
sum_of_p = np.zeros((2, 1))
temp_1  = seq[:,:,0]
frames = seq.shape[2]
temp = True
rects = []
for i in range(0, frames-1):
    if(temp):
        It = seq[:, :, i]
    It1 = seq[:, :, i+1]
    rect = np.array([min_x, min_y, max_x, max_y])
    rect = rect.reshape(4, 1)
    rects.append(rect[:, 0])
    p_new = LucasKanade(It, It1, rect, threshold, num_iters).reshape(2, 1)
    sum_of_p = sum_of_p + p_new
    p_prime = LucasKanade(temp_1, It1, rect_orig, threshold, num_iters, sum_of_p)
    if np.linalg.norm(sum_of_p - p_prime) < threshold: 
        sum_of_p = p_prime
        min_x = rect_orig[0] + sum_of_p[0, 0]
        max_x = rect_orig[2] + sum_of_p[0, 0]
        min_y = rect_orig[1] + sum_of_p[1, 0]
        max_y = rect_orig[3] + sum_of_p[1, 0]

    else: 
        sum_of_p = sum_of_p - p_new
        temp = False

rect = np.array([min_x, min_y, max_x, max_y])
rect = rect.reshape(4, 1)
rects.append(rect[:, 0])
np.save("carseqrects-wcrt.npy", rects)

rect_0 = np.load('./carseqrects.npy')

for i in range(0, frames -1):
    It = seq[:, :, i]
    rect = rects[i]
    if i + 1 == 1 or (i + 1)%100 == 0:
        fig, ax = plt.subplots(1)
        plt.axis('off')
        ax.imshow(It, cmap = 'gray')
        rect1 = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], linewidth = 1, edgecolor = 'r', fill = False)
        ax.add_patch(rect1)
        rect2 = rect_0[i]
        rect2 = patches.Rectangle((rect2[0], rect2[1]), rect2[2] - rect2[0], rect2[3] - rect2[1], linewidth = 1, edgecolor = 'b', fill = False)
        ax.add_patch(rect2)
        plt.savefig("correctedCarSeq" + str(i+1) + ".png")
        plt.show()


