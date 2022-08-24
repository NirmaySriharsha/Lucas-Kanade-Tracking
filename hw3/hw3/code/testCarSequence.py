import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("./data/carseq.npy")
frames = seq.shape[2]
rect = [59, 116, 145, 151]
width = rect[2] - rect[0]
height = rect[3] - rect[1]
rects = []
rect_copy =np.copy(rect)
rects.append(rect_copy)
for i in range(0, frames-1):
    print(i)
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]
    if i+1 == 1 or i+1 == 100 or i+1 == 200 or i+1 == 300 or i+1 == 400:
        fig, ax = plt.subplots(1)
        plt.axis('off')
        ax.imshow(It, cmap = 'gray')
        plot_rectangle = patches.Rectangle((rect[0], rect[1]), width, height, linewidth = 1, edgecolor = 'r', facecolor = 'none')
        ax.add_patch(plot_rectangle)
        plt.savefig("CarSeq" + str(i+1)+".png")
        plt.show()
    
    p = LucasKanade(It, It1, rect, threshold, num_iters)
    rect[0] = rect[0] + p[0]
    rect[1] = rect[1] + p[1]
    rect[2] = rect[2] + p[0]
    rect[3] = rect[3] + p[1]
    rects.append(np.copy(rect))

rects = np.array(rects)
np.save('./carseqrects.npy', rects)
