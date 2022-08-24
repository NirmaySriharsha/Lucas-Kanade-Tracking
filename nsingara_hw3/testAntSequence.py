import argparse
from re import S
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
import time

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-3, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.75, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance
time_total = 0
seq = np.load('./data/antseq.npy')
frames = seq.shape[2]
for i in range(frames-1):
    print(i)
    start = time.time()
    image1 = seq[:, :, i]
    image2 = seq[:, :, i+1]
    mask = SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance)
    end = time.time()
    time_total += end - start
    if (i+1)%30==0:
        plt.imshow(image2, cmap = 'gray')
        plt.axis('off')
        for r in range(mask.shape[0]-1):
            for c in range(mask.shape[1]-1):
                if mask[r, c] == 1:
                    plt.scatter(c, r, s = 1, c = 'b', alpha=0.5)
        #plt.savefig("AntSeq" + str(i+1) + ".png")
        plt.savefig("InvAntSeq" + str(i+1) +".png")
        plt.show()
print(time_total)