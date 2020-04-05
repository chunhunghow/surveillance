import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
import time
import re

def to_camera_coord(pts,h):
    
    homog = np.dot(h,np.append(pts,[1]))
    homog = homog / homog[-1]
    return homog[:-1].astype('uint32')

def to_camera_box(bbox_coord,h):
    assert len(bbox_coord) ==4
    (x1,y1) = to_camera_coord(bbox_coord[:2],h)
    (x2,y2) = to_camera_coord(bbox_coord[2:],h)
    return np.array([x1,y1,x2,y2])

def homography_mat(cam1,cam2):
    mat = np.array([])
    assert len(cam1) == len(cam2)
    for i in range(len(cam1)):
        c1 = cam1[i]
        c2 = cam2[i]
        row = np.array([[c2[0],c2[1], 1,0,0,0,-1*c2[0]*c1[0], -1*c1[0]*c2[1]],
                       [0,0,0, c2[0],c2[1], 1, -1*c2[0]*c1[1], -1*c1[1]*c2[1]]
                       ])
        #row2 is the bottom right x,y
#         row2 = np.array([[c2[2],c2[3], 1,0,0,0,-1*c2[2]*c1[2], -1*c1[2]*c2[3]],
#                        [0,0,0, c2[2],c2[3], 1, -1*c2[2]*c1[3], -1*c1[3]*c2[3]]
#                        ])
        if i ==0 :
            mat = row
            b = c1
        else:
            mat = np.append(mat,row, 0)
            b = np.append(b,c1)
            
    h_man = np.dot(np.linalg.inv(np.dot(mat.T,mat)),np.dot(mat.T,b))
    h_man = np.reshape(np.append(h_man,1),(3,3))
    return h_man


def homography_mat2(cam1,cam2):
    mat = np.array([])
    assert len(cam1) == len(cam2)
    for i in range(len(cam1)):
#         c1 = cam1[i]
#         c2 = cam2[i]
#         row = np.array([[c2[0],c2[1], 1,0,0,0,-1*c2[0]*c1[0], -1*c1[0]*c2[1]],
#                        [0,0,0, c2[0],c2[1], 1, -1*c2[0]*c1[1], -1*c1[1]*c2[1]]
#                        ])

#         if i ==0 :
#             mat = row
#             b = c1
#         else:
#             mat = np.append(mat,row, 0)
#             b = np.append(b,c1)
        c1 = cam1[i]
        c2 = cam2[i]
        row = np.array([[c2[0],c2[1], 1,0,0,0,-1*c2[0]*c1[0], -1*c1[0]*c2[1], -c1[0]],
                       [0,0,0, c2[0],c2[1], 1, -1*c2[0]*c1[1], -1*c1[1]*c2[1], -c1[1]]
                       ])
        if i ==0:
            mat = row
        else:
            mat = np.append(mat,row,0)
            
#     h_man = np.dot(np.linalg.inv(np.dot(mat.T,mat)),np.dot(mat.T,b))
#     h_man = np.reshape(np.append(h_man,1),(3,3))
    u,d,v = np.linalg.svd(np.dot(mat.T,mat))
    h = np.reshape(u[-1],(3,3))
    
    return h




def concat_video(args):

    if not args.display:
        assert (args.box1 is not None) & (args.box2 is not None) 
        bbox1 = np.load(args.box1,allow_pickle=True)
        bbox2  =np.load(args.box2,allow_pickle=True)
        stop_at = min(len(bbox1),len(bbox2))
        
        bbox1 = [arr[0][0] if arr is not None else None for arr in [l[0] for l in bbox1]]
        bbox2 = [arr[0][0] if arr is not None else None for arr in [l[0] for l in bbox2]]
        inters = [i  for i in range(stop_at) if (bbox1[i] is not None) & (bbox2[i] is not None)]
        #filter out only those frame where two cameras have correct detection on single person
        
        
        bbox1_train = [bbox1[i] for i in range(len(bbox1[:stop_at])) if i in inters]
        bbox2_train = [bbox2[i] for i in range(len(bbox2[:stop_at])) if i in inters]
        
        
        
        dis = lambda x : np.sqrt(x[0]**2 + x[1]**2 )
        m = np.mean(bbox1_train,axis=0)
        q = np.quantile([dis(arr)-dis(m) for arr in bbox1_train],0.99)
        inters2 = [i for i in range(len(bbox1_train)) if (dis(bbox1_train[i])-dis(m)) < q]
        #bbox1_train = np.array(bbox1_train)[np.array(inters2),:]
        #bbox2_train = np.array(bbox2_train)[np.array(inters2),:]
       
       

     #split (x1,y1) , (x2,y2) , for destination and source cameras, cam1 is source
        
        pts_src = np.append(np.array([arr[:2] for arr in bbox1_train]),np.array([arr[2:] for arr in bbox1_train]),axis=0)
        pts_dst = np.append(np.array([arr[:2] for arr in bbox2_train]),np.array([arr[2:] for arr in bbox2_train]),axis=0)
            

               #find homography
        if args.h_file is None:
            h,status = cv2.findHomography(pts_dst,pts_src,method=cv2.LMEDS,ransacReprojThreshold=1)
            np.save('npy/pts_{}'.format(args.box2.split('/')[-1][:-4]),np.array([pts_dst,pts_src]))
        else:
            print('Loading numpy weights')
            h = np.load(args.h_file)
        #h = homography_mat(pts_src,pts_dst)

        h_inv = np.linalg.inv(h)
        #pts_dst here is camera 1




    vid1 = args.videopath1
    vid2 = args.videopath2
    freeze = args.freeze
    forward = args.forward
    cap1 = cv2.VideoCapture(vid1)
    cap2 = cv2.VideoCapture(vid2)
    videos = [cap1,cap2]
    if forward is not None:
        assert ':' in forward
        forward_camera, forward_duration = forward.split(':')
        forward_camera = int(forward_camera) -1
        assert forward_camera <= len(videos), 'camera starts from 1 to k only'
        forward_duration = int(forward_duration)
        for i in range(forward_duration):
            videos[forward_camera].read()
    if freeze is not None:
        assert ':' in freeze
        freeze_camera, freeze_duration = forward.split(':')
        freeze_camera = int(freeze_camera) -1
        assert freeze_camera <= len(videos), 'camera starts from 1 to k only'
        freeze_duration = int(freeze_duration)

    writer = None
    output = None
    idx_frame = 0
    count = -1

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        idx_frame += 1
        ret1,frame1 = videos[0].read()
        ret2,frame2 = videos[1].read()
        if output is None:
            (h1,w1) = frame1.shape[:2]
            (h2,w2) = frame2.shape[:2]
            output = np.zeros((max(h1,h2),w1+w2,3),dtype="uint8")
        if idx_frame % args.frame_interval != 0:
            continue
        count += 1
        if not args.display:
            if count > stop_at:
                break

        output[:max(h1,h2),:w1,:] = frame1
        output[:max(h1,h2),w1:,:] = frame2
        if args.filename is not None:
            if writer is None:
                writer = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(output.shape[1],output.shape[0]),True)

        if not args.display:
            if bbox1[count] is None:
                pass
            else:
                cv2.rectangle(output,(bbox1[count][0],bbox1[count][1]),(bbox1[count][2],bbox1[count][3]),(255,0,0),2)
                
                projection_dst = to_camera_box(bbox1[count],h_inv)
                #cv2.rectangle(output,(projection_dst[0]+w1,projection_dst[1]),(projection_dst[2]+w1,projection_dst[3]),(0,0,255),2)
            
            if bbox2[count] is None:
                pass
            else:
                cv2.rectangle(output,(bbox2[count][0]+w1,bbox2[count][1]),(bbox2[count][2]+w1,bbox2[count][3]),(255,0,0),2)
                
                # draw projection from camera 2
                projection = to_camera_box(bbox2[count],h)
                cv2.rectangle(output,(projection[0],projection[1]),(projection[2],projection[3]),(0,0,255),2)
            
        

        cv2.imshow('test',output)
        #time.sleep(0.4)
        if args.filename is not None:
            writer.write(output)
    
    cap1.release()        
    cap2.release()
    cv2.destroyAllWindows()



def arg_parse():
    arg = argparse.ArgumentParser()
    arg.add_argument('videopath',type=str,nargs='+')
    arg.add_argument('--filename',type=str)
    arg.add_argument('--forward',type=str,default=None)
    arg.add_argument('--freeze',type=str,default=None)
    arg.add_argument('--display_only',action='store_true',dest='display')
    arg.add_argument('--frame_interval',type=int,default=20)
    arg.add_argument('--box1',type=str)
    arg.add_argument('--box2',type=str)
    arg.add_argument('--h_file',type=str)
    return arg.parse_args()

if __name__ == '__main__':
    arg = arg_parse()
    concat_video(arg)




#       cv2.imshow('Video1',frame)
 #       cv2.imshow('Video2',frame2)


