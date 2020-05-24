# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:44:45 2020

@author: mnagd
"""

import numpy as np
import cv2
import csv
import os 
# cwd = os.getcwd()
# os.chdir('C:/Users/mnagd/Desktop/ASU/2nd Semester/Perception in Robotics/Assignment 3/') 

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)    
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

left_image = cv2.imread('../../images/task_2/left_0.png')
right_image = cv2.imread('../../images/task_2/right_0.png')

def camera_params(filename):
    intrinsics=[]
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            intrinsics.append(row)   
    return intrinsics

params= camera_params('../../parameters/left_camera_intrinsics.csv')
left_intrinsics=np.reshape([params[1],params[2],params[3]],(3,3))
left_intrinsics=left_intrinsics.astype(float)
dist1=np.reshape([params[5]],(1,5))
dist1=dist1.astype(float)

params= camera_params('../../parameters/right_camera_intrinsics.csv')
right_intrinsics=np.reshape([params[1],params[2],params[3]],(3,3))
right_intrinsics=right_intrinsics.astype(float)
dist2=np.reshape([params[5]],(1,5))
dist2=dist2.astype(float)

def imgpoints(image):
    twod_imgpoints=[]
    objpoints = [] # 3d point in real world space
    img = image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        twod_imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    imgsize=gray.shape[::-1]
    return twod_imgpoints,imgsize,objpoints

left_imgpoints, imgsize, objpoints= imgpoints(left_image)
right_imgpoints, imgsize, objpoints= imgpoints(right_image)

retval, left_intrinsics, dist1, right_intrinsics, dist2, R, T, E, F =cv2.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints,left_intrinsics,dist1, right_intrinsics, dist2, imgsize, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
undistorted_left=cv2.undistortPoints(np.reshape(left_imgpoints,(54,1,2)),left_intrinsics,dist1)
undistorted_right=cv2.undistortPoints(np.reshape(right_imgpoints,(54,1,2)),right_intrinsics,dist2) 
r1= np.identity(3)
r2=np.dot(R,r1)
T1=np.transpose([[0,0,0]])
T2=np.dot(R,T1)+T

P1=np.asarray([[r1[0][0], r1[0][1],r1[0][2], T1[0][0]],[r1[1][0], r1[1][1],r1[1][2], T1[1][0]],[r1[2][0], r1[2][1],r1[2][2], T1[2][0]]])
P2=np.asarray([[r2[0][0], r2[0][1],r2[0][2], T2[0][0]],[r2[1][0], r2[1][1],r2[1][2], T2[1][0]],[r2[2][0], r2[2][1],r2[2][2], T2[2][0]]])
undistorted_left=np.transpose(np.reshape((undistorted_left),(54,2)))
undistorted_right=np.transpose(np.reshape((undistorted_right),(54,2)))

points4D=cv2.triangulatePoints(P1,P2,undistorted_left,undistorted_right)

rectify_scale = -1
points4D=np.transpose(points4D)
points4D=np.true_divide(points4D[:,:3], points4D[:,[-1]])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(points4D[:,0],points4D[:,1],points4D[:,2])
ax.plot3D(r1[:,0],r1[:,1],r1[:,2])
ax.plot3D(r2[:,0]+T2[:,0],r2[:,1]+T2[:,0],r2[:,2]+T2[:,0])
fig.savefig('../../output/task_2/3D.png',dpi=600, bbox_inches = "tight")

R1,R2,P1,P2,Q,roi1, roi2=cv2.stereoRectify(left_intrinsics,dist1, right_intrinsics, dist2,imgsize, R, T, alpha = rectify_scale)
left_maps = cv2.initUndistortRectifyMap(left_intrinsics, dist1, R1, P1, imgsize, cv2.CV_32FC1)
right_maps = cv2.initUndistortRectifyMap(right_intrinsics, dist2, R2, P2, imgsize, cv2.CV_32FC1)

undistorted_l = cv2.undistort(left_image, left_intrinsics, dist1)
undistorted_r = cv2.undistort(right_image, right_intrinsics, dist2)

left_img_remap = cv2.remap(left_image, left_maps[0],left_maps[1], cv2.INTER_LINEAR)
right_img_remap = cv2.remap(right_image, right_maps[0],right_maps[1], cv2.INTER_LINEAR)

import matplotlib.pyplot as plt
og1=left_image 
og1=cv2.cvtColor(og1, cv2.COLOR_RGB2BGR)
undistorted_l=cv2.cvtColor(undistorted_l, cv2.COLOR_RGB2BGR)   
left_img_remap=cv2.cvtColor(left_img_remap, cv2.COLOR_RGB2BGR)
og2=right_image 
og2=cv2.cvtColor(og2, cv2.COLOR_RGB2BGR)
undistorted_r=cv2.cvtColor(undistorted_r, cv2.COLOR_RGB2BGR)   
right_img_remap=cv2.cvtColor(right_img_remap, cv2.COLOR_RGB2BGR)
fig, ax = plt.subplots(3, 2)
ax1, ax2, ax3, ax4, ax5, ax6 = ax.flatten()
ax1.imshow(og1,interpolation=None)
ax2.imshow(og2,interpolation=None)
ax3.imshow(undistorted_l,interpolation=None)
ax4.imshow(undistorted_r,interpolation=None)
ax5.imshow(left_img_remap,interpolation=None)
ax6.imshow(right_img_remap,interpolation=None)
ax5.set_xlabel(' \n Original (top), undistorted but not rectified (middle), undistorted\n and rectified (bottom);for "left_0.png" and "right_0.png"', horizontalalignment='center')
#ax6.set_xlabel(' \n\n\n\n Original (top), undistorted but not rectified (middle), undistorted\n and rectified (bottom);for "right_2.png"', horizontalalignment='center')
plt.savefig('../../output/task_2/left_and_right.png',dpi=600, bbox_inches = "tight")
plt.show()

string=np.asarray(['R'])
np.savetxt('../../parameters/stereo_calibration.csv',string, delimiter=',',fmt='%s')
with open('../../parameters/stereo_calibration.csv', 'a', newline='') as file:
    np.savetxt(file, R, delimiter=',')
    np.savetxt(file,np.asarray(['T']),fmt='%s')
    np.savetxt(file, T, delimiter=",")
    np.savetxt(file,np.asarray(['E']),fmt='%s')
    np.savetxt(file, E, delimiter=",")
    np.savetxt(file,np.asarray(['F']),fmt='%s')
    np.savetxt(file, F, delimiter=",")
    
string=np.asarray(['R1'])
np.savetxt('../../parameters/stereo_rectification.csv',string, delimiter=',',fmt='%s')
with open('../../parameters/stereo_rectification.csv', 'a', newline='') as file:
    np.savetxt(file, R1, delimiter=',')
    np.savetxt(file,np.asarray(['R2']),fmt='%s')
    np.savetxt(file, R2, delimiter=",")
    np.savetxt(file,np.asarray(['P1']),fmt='%s')
    np.savetxt(file, P1, delimiter=",")
    np.savetxt(file,np.asarray(['P2']),fmt='%s')
    np.savetxt(file, P2, delimiter=",")
    np.savetxt(file,np.asarray(['Q']),fmt='%s')
    np.savetxt(file, Q, delimiter=",")
