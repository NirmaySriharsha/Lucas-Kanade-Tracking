import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
print(num_iters)
    
seq = np.load("./data/girlseq.npy")
rect = [280, 152, 330, 318]
frames = seq.shape[2]
width = rect[2] - rect[0]
height = rect[3] - rect[1]
rects = []
rect_copy =np.copy(rect)
rects.append(rect_copy)
for i in range(0, frames-1):
    print(i)
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]
    if i+1 == 1 or i+1 == 20 or i+1 == 40 or i+1 == 60 or i+1 == 80:
        fig, ax = plt.subplots(1)
        plt.axis('off')
        ax.imshow(It, cmap = 'gray')
        plot_rectangle = patches.Rectangle((rect[0], rect[1]), width, height, linewidth = 1, edgecolor = 'r', facecolor = 'none')
        ax.add_patch(plot_rectangle)
        plt.savefig("GirlSeq" + str(i+1)+".png")
        plt.show()
    
    p = LucasKanade(It, It1, rect, threshold, num_iters)
    rect[0] = rect[0] + p[0]
    rect[1] = rect[1] + p[1]
    rect[2] = rect[2] + p[0]
    rect[3] = rect[3] + p[1]
    rects.append(np.copy(rect))

rects = np.array(rects)
np.save('./girlseqrects.npy', rects)
