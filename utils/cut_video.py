
import cv2
import numpy
import argparse




def cut(args):
    idx_frame = -1
    cap = cv2.VideoCapture(args.videopath)
    writer = None
    rb = False
    if args.until is not None:
        rb = True
    while True:
        
        ret,frame = cap.read()
        if ret == True:
            idx_frame += 1
            if writer is None:
                if args.save_path is not None:
                    
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    writer = cv2.VideoWriter(args.save_path,fourcc,10,(frame.shape[1],frame.shape[0]))

            if idx_frame < args.cut_from:
                continue
            #write frame
            if args.save_path is not None:
                writer.write(frame)
            cv2.imshow('test',frame)
            
            if rb:
                if idx_frame >= args.until:
                    break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    if args.save_path is not None:
        writer.release()



def arg_parse():
    args = argparse.ArgumentParser()
    args.add_argument('videopath',type=str)
    args.add_argument('--save_path',type=str)
    args.add_argument('--cut_from',type=int,default = 0)
    args.add_argument('--until',type=int)
    return args.parse_args()



if __name__ == '__main__':
    arg = arg_parse()
    cut(arg)
