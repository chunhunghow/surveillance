import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
import time
import re
import logging

def concat_video(args): 
    vid = args.videopath
    freeze = args.freeze
    forward = args.forward
    caps = [cv2.VideoCapture(v) for v in vid]
    if forward is not None:
        assert all([':' in f for f in forward])
        forward_camera = [int(f.split(':')[0]) -1 for f in forward ]
        forward_duration = [int(f.split(':')[1]) for f in forward]
        assert all([f <= len(caps) for f in forward_camera]), 'camera starts from 1 to k only'
        for i,cam in enumerate(forward_camera):
            for j in range(forward_duration[i]):
                caps[cam].read()
    if freeze is not None:
        assert all([':' in f for f in freeze])
        freeze_camera = [int(f.split(':')[0]) -1 for f in freeze ]
        freeze_duration = [int(f.split(':')[1]) for f in freeze ]
        freeze_camera = int(freeze_camera) -1
        assert freeze_camera <= len(caps), 'camera starts from 1 to k only'
        freeze_duration = int(freeze_duration)

    writer = None
    count = 0
    store_scene = []
    store_shape = None
    output = None
    for i in range(len(caps)):
        ret,frame = caps[i].read()
        if ret:
            h,w = frame.shape[:2]
        store_scene += [(ret,frame)]

    assert (len(set([scene[1].shape[0] for scene in store_scene if scene[0]==True])) == 1) and ( len(set([scene[1].shape[1] for scene in store_scene if scene[0]==True])) == 1), 'All feeds should have same size , it would be expensive to scale video here'
    
    #first frame used to validation above

    idx_frame = 0
    logging.warning('Scale feeds resolution to 640x480')
    while True:
        idx_frame += 1
        if idx_frame % args.frame_interval != 0:
            continue
        store_scene = []
        #for i in range(len(caps)):
        #    ret,frame = caps[i].read()
        #    store_scene += [(ret,frame)]
        store_scene = [c.read() for c in caps]

        if all([scene[0] for scene in store_scene]):
            
            if len(caps) //2 == 1:
                if output is None:
                    output = np.zeros((h,w*len(caps),3),dtype="uint8")
                for i,scene in enumerate(store_scene):
                    output[:h,i*w:(i+1)*w,:] = scene[1]
            else:
                if output is None:
                    output = np.zeros((2*h,len(caps)//2 * w,3),dtype='uint8')
                for i in range(len(store_scene)//2):

                    output[:h,i*w:(i+1)*w,:] = store_scene[2*i][1]
                    output[h:,i*w:(i+1)*w,:] = store_scene[2*i +1][1]

        else:
            continue
        if args.savepath is not None:
            if writer is None:
                writer = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,output.shape[:2],True)
        cv2.imshow('test',output)
        
        #time.sleep(0.4)
        if args.savepath is not None:
            writer.write(output)
    
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap1.release()        
    cap2.release()
    cv2.destroyAllWindows()



def arg_parse():
    arg = argparse.ArgumentParser()
    arg.add_argument('videopath',type=str,nargs='+')
    arg.add_argument('--savepath',type=str)
    arg.add_argument('--forward',type=str,nargs='+')
    arg.add_argument('--freeze',type=str,nargs='+')
    arg.add_argument('--frame_interval',type=int,default=10)
    return arg.parse_args()
    return arg.parse_args()

if __name__ == '__main__':
    arg = arg_parse()
    concat_video(arg)




#       cv2.imshow('Video1',frame)
 #       cv2.imshow('Video2',frame2)


