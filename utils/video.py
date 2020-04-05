
import cv2
import numpy as np
import argparse
import time
def play_video(args):
    
    cap = cv2.VideoCapture(args.video_path)
    #cap.set(cv2.CAP_PROP_FPS,args.frame_rate)
    all_frame = []
    idx_frame = 0
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = None
    while True :
        if idx_frame % args.frame_interval != 0:
            continue

        ret, frame = cap.read()
        #all_frame += [frame]
        #for trans in list(args.__dict__.keys())[3:]:
        #    if dict(args.__dict__)[trans] is not None:
        #        frame = affine(frame, trans,args.__dict__[trans])
        #frame = cv2.resize(frame,  (200,150)) 
        if args.save_path is not None:
            if writer is None:
                writer = cv2.VideoWriter(args.save_path,fourcc,10,(frame.shape[1],frame.shape[0]))

            writer.write(frame)
        cv2.imshow("test",frame)
        #time.sleep(0.05)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def affine(frame,trans,value):
    mat = np.diag([1]*3)
    if trans == 'rotation':
        rot_mat = np.array([[np.cos(value *np.pi/180), -np.sin(value *np.pi/180),0],
        [np.sin(value *np.pi/180) , np.cos(value *np.pi/180),0],
        [0,0, 1]])
        mat = np.dot(mat,rot_mat)
    
    if trans == 'scale':
        #scale_mat = np.diag([value]*3)
        #mat = np.dot(mat,scale_mat)
        
        frame = cv2.resize(frame,  (200,150)) 
        return frame
    #mat[0,1] = 0.5
    #mat[1,0] = 0.7
    
    if trans == 'translation':
        trans_mat = np.diag([0]*3)
        trans_mat[0,-1] = value[0]
        trans_mat[1,-1] = value[1]
        mat = mat + trans_mat

    else:
        pass


    size = frame.shape
    new_frame = np.zeros(size)
    #new_frame = np.zeros((int(2*(np.sqrt((size[0]/2)**2 + size[1]/2)**2 + 5)) , int(2*(np.sqrt((size[0]/2)**2 + size[1]/2)**2 +5)),3))
    for i in range(len(frame)):
        for j in range(len(frame[i])):
            try:
                new_coord = np.dot(mat,np.array([i,j,1]))
                new_frame[int(new_coord[0]),int(new_coord[1]),:] = frame[i][j]
            except:
                pass


    return new_frame.astype('uint8')
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path' , type=str)
    parser.add_argument("--frame_rate",type=int , default=10)
    parser.add_argument("--frame_interval",type=int , default=20)
    parser.add_argument("--save_path",type=str)
    #parser.add_argument('--affine',action='store_true')
    parser.add_argument('--scale',type=int)
    parser.add_argument('--translation',type=int,nargs=2)
    parser.add_argument('--rotation',type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    play_video(args)
