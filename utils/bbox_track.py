from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import time
import argparse

'''just specify the path of npy file under tracking'''
'''only works for old bbox npy '''


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """

    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)



def show(args):
    coord = np.load(args.path,allow_pickle=True)


    #fig,ax = plt.subplots(1,facecolor='white')
    plt.figure()
    plt.ion()
    output = np.zeros((500,500,3),dtype='uint8') + 255
    idx = 0
    for bbox in coord:
        
        plt.cla()
        
        for i in bbox:
            coord,id_ = i
            x1,x2,y1,y2 = coord
            dot = [x1+(x2-x1)//2,y2]
            #dot = [y2,x1+(x2-x1)//2]
            output[dot[1]-4:dot[1]+4,dot[0]-4 : dot[0]+4,:] = compute_color_for_labels(id_) 
            if args.savedir is not None:
                if len(str(idx)) == 1:
                    print_idx = '00' + str(idx)
                elif len(str(idx)) == 2:
                    print_idx = '0' + str(idx)
                else:
                    print_idx = str(idx)
                plt.imsave(args.savedir + '/'+print_idx + '.png',output)
            idx +=1 

        plt.imshow(output)
        plt.show()
        plt.pause(0.05)
        #for b in bbox:
        #    if b is None:
        #        break
        #if bbox[0] is None:
        #    continue
        #(coord, id_) = bbox[0][0]
        #print(len(bbox[0]))
        x1,x2,y1,y2 = coord
            #rect=patches.Rectangle(xy=(x1,-y1),width=x2-x1,height=-(y2-y1),edgecolor='g', linewidth=3,fill=False)
            #ax.add_patch(rect)
        
        #output[dot[1],dot[0],:] = [255,0,0]
        #plt.scatter(x1+(x2-x1)/2,-y2)
        #plt.ylim((-500,0))
        #plt.xlim((0,600))
        #plt.imshow(output)
        #plt.show()
        #plt.pause(0.05)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("path",type=str)
    parser.add_argument('--savedir',type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arg()
    show(args)

