
import cv2
import numpy as np
import argparse



if __name__ == '__main__':
    parse_args = argparse.ArgumentParser()
    parse_args.add_argument('pts_path',type=str)
    parse_args.add_argument('savepath',type=str)
    args = parse_args.parse_args()
    pts = np.load(args.pts_path)
    pts_src=  []
    pts_dest = []
    for i in range(len(pts)//2):
        pts_src += [pts[2*i]]
        pts_dest += [pts[2*i+1]]
    h = cv2.findHomography(np.array(pts_dest),np.array(pts_src))[0]
    np.save(args.savepath,h)
