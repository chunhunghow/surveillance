

from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import cv2

class Viewpoint:
    '''Called from drawing space boundary'''
    
    
    def __init__(self,boundary):
        self.box = boundary
        self.pts = np.empty((0,2))
        y = [arr[1] for arr in boundary] #written in (x,y) not (y,x)
        self.view_cam = y.index(max(y))
        self.homography()
        self.distance = None

    def homography(self,scale=5):
        '''transform into a square , use scale in multiple of 100 just for ease of visualization , 
            but the ratio affects the eps used in dbscan

            All these only need to be done one frame, one time and store M 

            Now its POC
        '''
        self.scale = scale
        self.dst = np.array([[100,100],[100,100*scale],[100*scale,100*scale],[100*scale,100]])
        self.M = cv2.findHomography(self.box[:-1],self.dst)[0]
        self.viewpoint = self.dst[self.view_cam,:]
        
   


class Localize(Viewpoint):
    # set to be taking foot points as input first, later on can modify into bbox and compute foot point


    def add_point(self,pts):#id
        self.view_loc = np.array([[p[1],p[0]] for p in pts])
        self.pts = np.array([homo(arr,self.M) for arr in self.view_loc])
        self.distance = np.sqrt(np.sum((self.pts - self.viewpoint)**2,axis=1))
    

        #self.id = id
    def adjust_err(self,magnitude=0.1):
        self.direction = self.viewpoint - self.pts
        self.pts = np.exp(magnitude *-1* self.direction) + self.pts
    
    
    def add_label(self,label):
        self.label = label

    def scene_affine(self,im):
        self.view_image = cv2.warpPerspective(im,self.M,dsize=(im.shape[0],im.shape[1]))
        
     
        
# a camera is a class, takes in 4 input of space boundary,all have fix square for homography 



class cluster():
    def __call__(self,views):
        self.allpts = np.empty((0,2))
        self.dist_to_cam = np.empty((0,1))
        self.views = views
        for v in views:
            self.allpts = np.append(self.allpts,v.pts,axis=0)
            self.dist_to_cam = np.append(self.dist_to_cam,v.distance) if v.distance is not None else np.append(self.dist_to_cam,np.empty((0,1)))
        
        self.clustering = DBSCAN(eps=v.scale*10, min_samples=1).fit(self.allpts)
        self.labels = self.clustering.labels_
    def update(self):
        '''we append all the points according to view sequence as allpts, so we update them in same sequence'''
        labels = self.clustering.labels_
        for v in self.views:
            v.add_label(labels[:len(v.pts)])
            labels = labels[len(v.pts):]
        
    def return_centric(self):
        norm_loc = np.empty((0,2))
        for ele in set(self.clustering.labels_):
            tmp_pts = self.allpts[self.clustering.labels_ == ele,:]
            tmp_dis = np.expand_dims(self.dist_to_cam[self.clustering.labels_ == ele],1)

            norm_loc = np.append(norm_loc,np.expand_dims(np.sum(tmp_pts * tmp_dis / tmp_dis.sum(),axis=0),axis=0) ,axis=0)

    
        return norm_loc    





 
def homo(pts,h):
    pts = np.append(pts,[1])
    new = np.dot(h,pts)
    new = new / new[-1]
    return new[:-1]

