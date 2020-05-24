# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:57:29 2020

@author: mnagd
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import csv
import os 
# cwd = os.getcwd()
# os.chdir('C:/Users/mnagd/Desktop/ASU/2nd Semester/Perception in Robotics/Assignment 3/') 

left=cv2.imread('../../images/task_3_and_4/left_5.png')
right=cv2.imread('../../images/task_3_and_4/right_5.png')
gray_l=cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
gray_r=cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)

def camera_params(filename):
    params=[]
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            params.append(row)
    return params

params=camera_params('../../parameters/stereo_calibration.csv')
R=np.reshape([params[1],params[2],params[3]],(3,3))
T=np.reshape([params[5],params[6],params[7]],(3,1))
E=np.reshape([params[9],params[10],params[11]],(3,3))
F=np.reshape([params[13],params[14],params[15]],(3,3))
R=R.astype('float64')
T=T.astype('float64')
E=E.astype('float64')
F=F.astype('float64')

params=camera_params('../../parameters/stereo_rectification.csv')
R1=np.reshape([params[1],params[2],params[3]],(3,3))
R2=np.reshape([params[5],params[6],params[7]],(3,3))
P1=np.reshape([params[9],params[10],params[11]],(3,4))
P2=np.reshape([params[13],params[14],params[15]],(3,4))
Q=np.reshape([params[17],params[18],params[19],params[20]],(4,4))
R1=R1.astype('float64')
R2=R2.astype('float64')
P1=P1.astype('float64')
P2=P2.astype('float64')
Q=Q.astype('float64')

params= camera_params('../../parameters/right_camera_intrinsics.csv')
right_intrinsics=np.reshape([params[1],params[2],params[3]],(3,3))
dist2=np.reshape([params[5]],(1,5))
right_intrinsics=right_intrinsics.astype('float64')
dist2=dist2.astype('float64')

left_intrinsics=np.reshape([params[1],params[2],params[3]],(3,3))
dist1=np.reshape([params[5]],(1,5))
left_intrinsics=left_intrinsics.astype('float64')
dist1=dist1.astype('float64')

(x,y) = gray_l.shape[:2]
left_mapx,left_mapy = cv2.initUndistortRectifyMap(left_intrinsics, dist1, R1, P1,(y,x),cv2.CV_32FC1)
right_mapx,right_mapy = cv2.initUndistortRectifyMap(right_intrinsics, dist2, R2, P2,(y,x),cv2.CV_32FC1)
left=cv2.remap(gray_l, left_mapx, left_mapy, cv2.INTER_LINEAR)
right=cv2.remap(gray_r, right_mapx, right_mapy, cv2.INTER_LINEAR)

fx = 323.3818        
baseline = 62   
disparities = 48 
block = 17        

sbm = cv2.StereoBM_create(numDisparities=disparities,
                          blockSize=block)

disparity = sbm.compute(left, right)
valid_pixels = disparity > 0

plt.figure(1)
plt.imshow(disparity,'gray')
plt.show()

Z=(fx*baseline)/disparity
plt.figure(2)
plt.imshow(Z,'gray')
plt.show()
left=cv2.imread('../../images/task_3_and_4/left_5.png')
right=cv2.imread('../../images/task_3_and_4/right_5.png')
left=cv2.remap(left, left_mapx, left_mapy, cv2.INTER_LINEAR)
right=cv2.remap(right, right_mapx, right_mapy, cv2.INTER_LINEAR)
left=cv2.cvtColor(left, cv2.COLOR_RGB2BGR)   
right=cv2.cvtColor(right, cv2.COLOR_RGB2BGR)  
fig, ax = plt.subplots(1, 3)
ax1, ax2, ax3 = ax.flatten()
ax1.imshow(left,interpolation=None)
ax2.imshow(right,interpolation=None)
ax3.imshow(disparity,interpolation=None)
ax2.set_xlabel('\nRectified images from the stereo camera (left)\n and the disparity map (right).', size=5, horizontalalignment='center')
plt.savefig('../../output/task_4/disparity_img1.png',dpi=600, bbox_inches = "tight")
plt.show()

fig, ax = plt.subplots(1, 3)
ax1, ax2, ax3 = ax.flatten()
ax1.imshow(left,interpolation=None)
ax2.imshow(right,interpolation=None)
ax3.imshow(Z,interpolation=None)
ax2.set_xlabel('\nRectified images from the stereo camera (left)\n and the depth map (right).', size=5, horizontalalignment='center')
plt.savefig('../../output/task_4/depth_img2.png',dpi=600, bbox_inches = "tight")
plt.show()

