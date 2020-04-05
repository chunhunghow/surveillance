import time
import cv2
import numpy as np
import argparse

def annotate(args):
    img1 = cv2.imread(args.scene1)
    img2 = cv2.imread(args.scene2)
    if img1.shape[0] > 1200:
        img1 = cv2.resize(img1,(640,480))
    if img2.shape[0] > 1200:
        img2 = cv2.resize(img2,(640,480))

    (h1,w1) = img1.shape[:2]
    img1 = cv2.putText(img1,'1',fontScale=2,org=(50,50),fontFace=cv2.FONT_HERSHEY_PLAIN,color=(0,0,255),thickness=2)
    (h2,w2) = img2.shape[:2]
    img2 = cv2.putText(img2,'2',fontScale=2,org=(50,50),fontFace=cv2.FONT_HERSHEY_PLAIN,color=(0,0,255),thickness=2)
    output = np.zeros((max(h1,h2),w1+w2,3),dtype="uint8")
    output[:max(h1,h2),:w1,:] = img1
    output[:max(h1,h2),w2:,:] = img2
    click_times = 0
    x_coord = None
    y_coord =None 
    xy = []

    def clicks(event,x,y,flags,param):
        nonlocal click_times
        nonlocal x_coord
        nonlocal y_coord
        nonlocal xy
        if event == cv2.EVENT_LBUTTONDOWN:
            x_coord = x
            y_coord = y
            click_times += 1
            if click_times % 2 ==0:
                xy += [[x-w1,y]]
            else:
                xy += [[x,y]]
        elif event== cv2.EVENT_LBUTTONUP:
            pass
    out_temp = output.copy()
    cv2.namedWindow('image')
    #cv2.setMouseCallback('image',clicks,[x_coord,y_coord,click_times])
    cv2.setMouseCallback('image',clicks)
    while True:
        
        store_for_undo = out_temp.copy()
        if click_times %2 == 0:
            out_temp = cv2.putText(out_temp,'1',fontScale=2,org=(50,50),fontFace=cv2.FONT_HERSHEY_PLAIN,color=(0,255,0),thickness=2)

            out_temp = cv2.putText(out_temp,'2',fontScale=2,org=(w1+50,50),fontFace=cv2.FONT_HERSHEY_PLAIN,color=(0,0,255),thickness=2)
        else:
            
            out_temp = cv2.putText(out_temp,'1',fontScale=2,org=(50,50),fontFace=cv2.FONT_HERSHEY_PLAIN,color=(0,0,255),thickness=2)
            out_temp = cv2.putText(out_temp,'2',fontScale=2,org=(w1+50,50),fontFace=cv2.FONT_HERSHEY_PLAIN,color=(0,255,0),thickness=2)

        cv2.imshow('image',out_temp)
        if click_times > 0:

            out_temp[y_coord-3 :y_coord+3,x_coord-3 : x_coord +3,:] = [0,0,255]
        if cv2.waitKey(1) & 0xFF == ord('q'):
        
            cv2.destroyAllWindows()
            break
    #in progress
        if cv2.waitKey(1) & 0xFF == ord('z'):
            xy = xy[:-1]
            click_times -= 1

            out_temp = store_for_undo

            cv2.imshow('image1',out_temp)
            print('Undo')
            continue



    if args.savepath is not None:
        np.save(args.savepath,xy)
    return
    

def click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0] = x
        param[1] = y
        #param[-1] += 1
        click_times += 1

    elif event== cv2.EVENT_LBUTTONUP:
        pass

    
            
        




def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('scene1',type=str)
    args.add_argument('scene2',type=str)
    args.add_argument('--savepath',type=str)
    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    annotate(args)

