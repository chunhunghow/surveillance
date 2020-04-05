import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
import time

plt.figure()
plt.ion()
H = np.load('npy/cam1_2.npy')

cam1 = np.load('npy/cam102_trunc_202.npy',allow_pickle=True)
cam2 = np.load('npy/cam2_02.npy',allow_pickle=True)


for i in range(min(len(cam1),len(cam2))):
    plt.cla()
    if (cam1[i][0] is None) or (cam2[i][0] is None):
        continue
    else:
        bbox1 = cam1[i][0][0][0]
        bbox2 = cam2[i][0][0][0]
        w = lambda b: b[2] - b[0]
        h = lambda b: b[3] - b[1]
        mid1 = [bbox1[0] + w(bbox1)//2,bbox1[1] + h(bbox1)//2]
        mid2 = [bbox2[0] + w(bbox2)//2,bbox2[1] + h(bbox2)//2]
        proj_mid2 = np.dot(H, np.array(mid2 + [1]))
        
        plot_pts = np.array([np.array(mid1),proj_mid2[:-1]])
        plt.scatter(plot_pts[:,0],plot_pts[:,1])
        plt.show()
        plt.pause(0.05)
