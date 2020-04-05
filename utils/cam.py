
import cv2
import numpy
import argparse

def play_video():
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FPS,fr)
    #all_frame = []
    while cap.grab():

        ret, frame = cap.read()
        #all_frame += [frame]
        cv2.imshow("test",frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH" , type=str)
    parser.add_argument("frame_rate",type=int , default=10)
    return parser.parse_args()


if __name__ == '__main__':
    play_video()
    #args = parse_args()
    #play_video(args.VIDEO_PATH,args.frame_rate)
