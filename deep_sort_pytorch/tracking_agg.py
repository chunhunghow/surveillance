import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
import time
import re
import logging
from yolov3_deepsort3 import *
from detector import build_detector
import time
from localization import *




def concat_video(scenes): 
    output = None
    
    assert (len(set([scene[1].shape[0] for scene in scenes])) == 1) and ( len(set([scene[1].shape[1] for scene in scenes ])) == 1), 'All feeds should have same size , it would be expensive to scale video here'
   
    h,w,_ = scenes[0].shape

    if np.ceil(len(scenes) /2) == 1:
        if output is None:
            output = np.zeros((h,w*len(scenes),3),dtype="uint8")
        for i,scene in enumerate(scenes):
            output[:h,i*w:(i+1)*w,:] = scene
    else:
        if output is None:
            output = np.zeros((2*h,np.ceil(len(scenes)/2).astype(int) * w,3),dtype='uint8')
        for i in range(np.ceil(len(scenes)/2).astype(int)):
            try:
                output[:h,i*w:(i+1)*w,:] = scenes[2*i]
            except:
                pass
            try:
                output[h:,i*w:(i+1)*w,:] = scenes[2*i +1]
            except:
                pass
    

    return output




def multisync(arg):
    cam_obj = []
    count = 0
    idx_frame = -1
    vids = args.VIDEO_PATH
    detector = build_detector(cfg,use_cuda= True) 

            #writer = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,output.shape[:2],True)


'''Assignment of view boundary file for homography'''
    if arg.calib is not None:
        boundary_file = np.load(arg.calib)
        assert len(boundary_file) == len(vids) , 'No of views and input calibration file are different'
    else:
        #arg.calib here is just the coord of the boundary view, right now we always input boundary file that means camera
        # is always calibrated, if not provided then just show independent detection and tracking
        boundary_file = [None]*len(vids)



'''Assign individual video scene into a individual class'''

    for i in range(len(vids)):
        arg.VIDEO_PATH = vids[i]
        
        #here I will assign each video scene into a class
        with VideoTracker(cfg,arg,detector,boundary_file[i],i+1) as vdo_trk:
            cam_obj += [vdo_trk]




'''film start rolling'''


    while count < 100:

        start = time.time()
        idx_frame += 1
        if idx_frame % args.frame_interval != 0:
            [c.run_scene(skip=True) for c in cam_obj] # dont process and skip the scene
            continue
        


        # process the scene, when run_scene is called, each detection in the scene will have its viewpoint noted down
        tracking_output = [c.run_scene() for c in cam_obj]


        if (args.calib is not None) and (any([len(out['vp'].pts)>0 for out in tracking_output])):
            #scene boundary file is provided


            cl = cluster() 
            cl([out['vp'] for out in tracking_output]) #cl is an independent class that only help to cluster the different viewpoints
            cl.update()

        

               #here output footpoint 

            for i,v in enumerate(tracking_output):
                plt.scatter(v['vp'].pts[:,0],v['vp'].pts[:,1]*-1,color='C'+str(i))
            centric = cl.return_centric()
            #plt.scatter(centric[:,0],centric[:,1]*-1,color='black')
            plt.savefig('../temp.png')
            plt.figure()
            col = np.array(['C' + str(i) for i in range(20)])
            plt.scatter(cl.allpts[:,0],cl.allpts[:,1]*-1,color=col[cl.labels])
            #plt.scatter(centric[:,0],centric[:,1]*-1,color='black')
            plt.savefig('../temp2.png')
            break




            for i,out in enumerate(tracking_output):
                if len(out['b']) > 0:
                    out['x'] = draw_boxes(out['x'],out['b'],out['vp'].label)
                    cv2.putText(out['x'],text=f"Number of people {len(out['b'])}",org=(out['x'].shape[1]-200,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX ,fontScale=0.6,thickness=2,color=(0,0,255)) 
        





        
                #if boundary file not provided, just normal multiple scenes indpdnt. detection
        else:
            for out in tracking_output:
                if len(out['b']) > 0:
                    out['x'] = draw_boxes(out['x'],out['b'],out['i'])
                    cv2.putText(out['x'],text=f"Number of people {len(out['b'])}",org=(out['x'].shape[1]-200,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX ,fontScale=0.6,thickness=2,color=(0,0,255)) 
 




        if arg.calib is not None:
            for i,out in enumerate(tracking_output):
                x1,x2 = out['vp'].dst[0] , out['vp'].dst[2]
                #bg = np.zeros((x2[0]-x1[0],x2[1]-x1[1],3)) + 255
                out['vp'].scene_affine(out['x'])
                bg = out['vp'].view_image
                for pts in out['vp'].pts:
                    try:
                        #print(pts,i)
                        bg[int(pts[1])-5:int(pts[1])+5,int(pts[0])-5:int(pts[0])+5,:] = [0,0,255]
                    except:
                        pass
                bg = cv2.resize(bg,(out['x'].shape[1] , out['x'].shape[0] ))

                tracking_output[i]['x'] = np.append(out['x'],bg, axis=1)




        #show the scene here

        for out in tracking_output:
        #    out['x'] = cv2.warpPerspective(out['x'],out['vp'].M,dsize=(out['x'].shape[0],out['x'].shape[1]))
            out['x'] = cv2.resize(out['x'],(500,300))    

                      
        #concat them 

        output = concat_video([out['x'] for out in tracking_output])
        cv2.imshow('test',output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            cv2.destroyAllWindows()

        count += 1

    

















if __name__ == '__main__':
    #os.chdir(os.path.dirname(os.path.realpath(__file__)))
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    #concat_video(arg)
    multisync(args)





