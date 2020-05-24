# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:16:26 2020

@author: mnagd
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os 
# cwd = os.getcwd()
# os.chdir('C:/Users/mnagd/Desktop/ASU/2nd Semester/Perception in Robotics/Assignment 3/') 

def camera_calibration(cam_image,x):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objp=objp
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    if x==0:
        images = glob.glob('../../images/task_1/left_*.png')
    elif x==1:
        images = glob.glob('../../images/task_1/right_*.png') 
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
            if fname=='../../images/task_1\\left_2.png':
                plotimg=img
            if fname=='../../images/task_1\\right_2.png':
                plotimg=img
            cv2.imshow('img',img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    img = cam_image
    # undistort
    (x,y) = img.shape[:2]
    mapx,mapy = cv2.initUndistortRectifyMap(mtx, dist, None, None,(y,x),cv2.CV_32FC1)
    undistored_image=cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    return x,mtx,dist,undistored_image,plotimg

x=0;
cam_image=cv2.imread('../../images/task_1/left_2.png')
x,mtx1,dist1,undistored_image1,plotimg1=camera_calibration(cam_image,x);
mtx1=np.asarray(mtx1)

string=np.asarray(['Left camera intrinsic matrix'])
np.savetxt('../../parameters/left_camera_intrinsics.csv',string, delimiter=',',fmt='%s')
with open('../../parameters/left_camera_intrinsics.csv', 'a', newline='') as file:
    np.savetxt(file, mtx1, delimiter=',')
    np.savetxt(file,np.asarray(['Left camera distortion co-efficients']),fmt='%s')
    np.savetxt(file, dist1, delimiter=",")
    
x=1;
cam_image=cv2.imread('../../images/task_1/right_2.png')
x,mtx2,dist2,undistored_image2,plotimg2=camera_calibration(cam_image,x);
mtx2=np.asarray(mtx2)

string=np.asarray(['Right camera intrinsic matrix'])
np.savetxt('../../parameters/right_camera_intrinsics.csv',string, delimiter=',',fmt='%s')
with open('../../parameters/right_camera_intrinsics.csv', 'a', newline='') as file:
    np.savetxt(file, mtx2, delimiter=',')
    np.savetxt(file,np.asarray(['Right camera distortion co-efficients']),fmt='%s')
    np.savetxt(file, dist2, delimiter=",")

og1=cv2.imread('../../images/task_1/left_2.png') 
og1=cv2.cvtColor(og1, cv2.COLOR_RGB2BGR)
plotimg1=cv2.cvtColor(plotimg1, cv2.COLOR_RGB2BGR)   
undistored_image1=cv2.cvtColor(undistored_image1, cv2.COLOR_RGB2BGR)
fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
ax1.imshow(og1,interpolation=None)
ax2.imshow(plotimg1,interpolation=None)
ax3.imshow(undistored_image1,interpolation=None)
ax2.set_xlabel(' \n The original calibration board pattern (left), the same pattern with\n corner points annotation (middle),and the undistorted pattern (right); using "left_2.png"')
plt.show()
plt.savefig('../../output/task_1/left_2.png',dpi=600, bbox_inches = "tight")

og2=cv2.imread('../../images/task_1/right_2.png')    
og2=cv2.cvtColor(og2, cv2.COLOR_RGB2BGR)
plotimg2=cv2.cvtColor(plotimg2, cv2.COLOR_RGB2BGR)   
undistored_image2=cv2.cvtColor(undistored_image2, cv2.COLOR_RGB2BGR)
fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
ax1.imshow(og2,interpolation=None)
ax2.imshow(plotimg2,interpolation=None)
ax3.imshow(undistored_image2,interpolation=None)
ax2.set_xlabel(' \n The original calibration board pattern (left), the same pattern with\n corner points annotation (middle), and the undistorted pattern (right); using "right_2.png"')
plt.show()
plt.savefig('../../output/task_1/right_2.png',dpi=600, bbox_inches = "tight")

