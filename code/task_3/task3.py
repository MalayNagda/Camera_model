# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:43:47 2020

@author: mnagd
"""

import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import os 
from scipy.spatial.kdtree import KDTree
# cwd = os.getcwd()
# os.chdir('C:/Users/mnagd/Desktop/ASU/2nd Semester/Perception in Robotics/Assignment 3/') 

def camera_params(filename):
    params=[]
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # read file row by row
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

params= camera_params('../../parameters/left_camera_intrinsics.csv')
left_intrinsics=np.reshape([params[1],params[2],params[3]],(3,3))
dist1=np.reshape([params[5]],(1,5))
left_intrinsics=left_intrinsics.astype('float64')
dist1=dist1.astype('float64')

params= camera_params('../../parameters/right_camera_intrinsics.csv')
right_intrinsics=np.reshape([params[1],params[2],params[3]],(3,3))
dist2=np.reshape([params[5]],(1,5))
right_intrinsics=right_intrinsics.astype('float64')
dist2=dist2.astype('float64')

img1=cv2.imread('../../images/task_3_and_4/left_9.png')
img2=cv2.imread('../../images/task_3_and_4/right_9.png')
(x,y) = img1.shape[:2]
left_mapx,left_mapy = cv2.initUndistortRectifyMap(left_intrinsics, dist1, None, None,(y,x),cv2.CV_32FC1)
right_mapx,right_mapy = cv2.initUndistortRectifyMap(right_intrinsics, dist2, None, None,(y,x),cv2.CV_32FC1)

def orb(img):
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.remap(gray, left_mapx, left_mapy, cv2.INTER_LINEAR)
    kps = orb.detect(gray,None)
    kps, des = orb.compute(gray, kps)
    # draw only keypoints location,not size and orientation
    keys = cv2.drawKeypoints(img,kps,color=(0,255,0), flags=0, outImage = None)
    plt.figure()
    plt.title('Keypoints detected')
    plt.imshow(keys),plt.show()
    return keys,kps,des,gray

def KDT_NMS(kps, descs, r=6, k_max=50):
    neg_responses = [-kp.response for kp in kps]
    order = np.argsort(neg_responses)
    kps = np.array(kps)[order].tolist()

    data = np.array([list(kp.pt) for kp in kps])
    kd_tree = KDTree(data)

    N = len(kps)
    removed = []
    for i in range(N):
        if i in removed:
            continue

        dist, inds = kd_tree.query(data[i,:],k=k_max,distance_upper_bound=r)
        for j in inds: 
            if j>i:
                removed.append(j)

    kp_filtered = [kp for i,kp in enumerate(kps) if i not in removed]
    descs_filtered = None
    if descs is not None:
        descs = descs[order]
        descs_filtered = np.array([desc for i,desc in enumerate(descs) if i not in removed],dtype=np.float32)
    return kp_filtered, descs_filtered

keys1, kps1, des1,gray = orb(img1)
cv2.imshow('gray1',gray)
keys2, kps2, des2,gray = orb(img2)
cv2.imshow('gray2',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

kp_filtered1, descs_filtered1= KDT_NMS(kps1, descs=des1, r=6, k_max=50)
keys = cv2.drawKeypoints(img1,kp_filtered1,color=(0,255,0), flags=0, outImage = None)
plt.figure()
plt.title('Keypoints detected ANMS on img1')
plt.imshow(keys),plt.show()

kp_filtered2, descs_filtered2= KDT_NMS(kps2, descs=des2, r=6, k_max=50)
keys2 = cv2.drawKeypoints(img2,kp_filtered2,color=(0,255,0), flags=0, outImage = None)
plt.figure()
plt.title('Keypoints detected using supression on img 2')
plt.imshow(keys2),plt.show()


bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
good = []
for match in matches:
    if match[0].distance < 0.7*match[1].distance:
        good.append(match[0])
'''draw_params = dict(matchColor = (0,255,0),singlePointColor = None,flags = 2)
img3 = cv2.drawMatches(img1,kps1,img2,kps2,good,None,**draw_params)
plt.figure()
plt.imshow(img3)
plt.title('Featured detected')
plt.show()'''

left_pts=[]
right_pts=[]
for m in good:
        left_pts.append(kps1[m.queryIdx].pt)
        right_pts.append(kps2[m.trainIdx].pt)
        
left_pts=np.transpose(left_pts)
right_pts=np.transpose(right_pts)
points4D=cv2.triangulatePoints(P1,P2,left_pts,right_pts)
points4D=np.transpose(points4D)
points4D=np.true_divide(points4D[:,:3], points4D[:,[-1]])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
fig = plt.figure(figsize=plt.figaspect(0.5))

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.scatter3D(points4D[:,0],points4D[:,1],points4D[:,2])

#fig, ax = plt.subplots(1, 2)
#ax1, ax2 = ax.flatten()
ax = fig.add_subplot(1, 3, 1)
ax.imshow(img1,interpolation=None)
ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.scatter3D(points4D[:,0],points4D[:,1],points4D[:,2])
ax.set_title('\n sparse depth triangulation results: (left) the original image, (middle) \nobtained 3D points, (right) 3D points on a cylinder surface from a certain view.', size=10, horizontalalignment='center')
ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.scatter3D(points4D[:,0],points4D[:,1],points4D[:,2])
plt.savefig('../../output/task_3/sparse_depth.png',dpi=600, bbox_inches = "tight")
plt.show()


bf = cv2.BFMatcher()
matches = bf.knnMatch(descs_filtered1,descs_filtered2, k=2)
good = []
for match in matches:
    if match[0].distance < 0.7*match[1].distance:
        good.append(match[0])
draw_params = dict(matchColor = (0,255,0),singlePointColor = None,flags = 2)
img3 = cv2.drawMatches(img1,kp_filtered1,img2,kp_filtered2,good,None,**draw_params)
plt.figure()
plt.imshow(img3)
plt.title('Featured detected')
plt.show()

left_pts=[]
right_pts=[]
for m in good:
        left_pts.append(kp_filtered1[m.queryIdx].pt)
        right_pts.append(kp_filtered2[m.trainIdx].pt)

fig, ax = plt.subplots(1, 2)
ax1, ax2 = ax.flatten()
ax1.imshow(keys1,interpolation=None)
ax2.imshow(keys,interpolation=None)
ax2.set_xlabel('\n Detected feature points (left) and selected \nlocal maxima feature points with a six pixel radius (right).', size=5, horizontalalignment='center')
plt.savefig('../../output/task_3/feature_points.png',dpi=600, bbox_inches = "tight")
plt.show()


fig, ax = plt.subplots(1)
ax.imshow(img3,interpolation=None)
ax.set_xlabel('\n Selected matches of feature points on the two views.', size=5, horizontalalignment='center')
plt.savefig('../../output/task_3/matched_points.png',dpi=600, bbox_inches = "tight")
plt.show()
