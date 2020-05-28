import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
import numpy as np
import cv2
import glob
import csv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#offset matrix
offset_matrix = np.array([[1, 0,      -1],
 [  0,     1, -1],
 [  0,          0,          1        ]])

def camera_params(filename):
    params=[]
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # read file row by row
        for row in reader:
            params.append(row)
    return params

params= camera_params('../../parameters/left_camera_intrinsics.csv')
left_intrinsics=np.reshape([params[1],params[2],params[3]],(3,3))
dist1=np.reshape([params[5]],(1,5))
left_intrinsics= left_intrinsics.astype('float64')
dist1=np.asarray(dist1.astype('float64'))

params= camera_params('../../parameters/right_camera_intrinsics.csv')
right_intrinsics=np.reshape([params[1],params[2],params[3]],(3,3))
dist2=np.reshape([params[5]],(1,5))
right_intrinsics=right_intrinsics.astype('float64')
dist2=dist2.astype('float64')

#This method to undistort the images
def image_undistort(img, mtx, dist):
 h, w = img.shape[:2]
 newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
 #undistoting images
 dst = cv2.undistort(img, mtx, dist, None, mtx)
 return dst

def right_angle_distance(img_pt_1, img_pt_2):
 return math.sqrt((img_pt_2[0] - img_pt_1[0]) ** 2 + (img_pt_2[1] - img_pt_1[1]) ** 2)

def orb(img):
    orb = cv2.ORB_create()
    kps = orb.detect(img,None)
    kps, des = orb.compute(img, kps)
    return kps,des

def plot_pyramid(axis, R, T):
 #print ("plotting pyramid")
 v = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [0, 0, 0]])
 v = v.T
 v = np.append(v, [[1, 1, 1, 1, 1]], axis=0)
 H = np.hstack((R, T))
 H = np.append(H, [[0, 0, 0, 1]], axis=0)
 v_t = (np.matrix(H) * np.matrix(v))
 v_t = np.delete(v_t, 3, 0)
 v = np.array(v_t.T)*3

 verts = [[v[0], v[1], v[4]], [v[0], v[3], v[4]],
          [v[2], v[1], v[4]], [v[2], v[3], v[4]], [v[0], v[1], v[2], v[3]]]

 axis.add_collection3d(Poly3DCollection(verts,
                                        facecolors='white', linewidths=1, edgecolors='k', alpha=.25))

#left and right images
left_image1=cv2.imread("../../images/task_7/left_1.png")
left_image1 = cv2.cvtColor(left_image1, cv2.COLOR_BGR2GRAY)

left_image2 = cv2.imread("../../images/task_7/left_3.png")
left_image2 = cv2.cvtColor(left_image2, cv2.COLOR_BGR2GRAY)

#distort the images
img1 = image_undistort(left_image1, left_intrinsics, dist1)
img2 = image_undistort(left_image2, left_intrinsics, dist1)

kp1, des1 = orb(img1)
kp2, des2 = orb(img2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                      flags=2)

cv2.imshow('img3',img3)
cv2.imwrite("../../output/task_7/Match.jpg",img3)

cv2.waitKey(1000)
cv2.destroyAllWindows()

left_image_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,2)

right_image_points = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1,2)

kp1_match = np.array([kp1[mat.queryIdx].pt for mat in matches])
kp2_match = np.array([kp2[mat.trainIdx].pt for mat in matches])

kp1_match_ud = cv2.undistortPoints(np.expand_dims(kp1_match, axis=1), left_intrinsics, dist1)
kp2_match_ud = cv2.undistortPoints(np.expand_dims(kp2_match, axis=1), left_intrinsics, dist1)

E, mask_e = cv2.findEssentialMat(kp1_match_ud, kp2_match_ud, focal=1.0, pp=(0., 0.),
                               method=cv2.RANSAC, prob=0.999, threshold=0.001)

mask_bool=mask_e.astype(bool)

imgx = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                      matchesMask=mask_bool.ravel().tolist(),flags=2)

cv2.imshow("inliners", imgx)
cv2.imwrite("../../output/task_7/Inliners.jpg",imgx)
cv2.waitKey(1000)
cv2.destroyAllWindows()

#Recover the required pose
points, R, t, mask1 = cv2.recoverPose(E, kp1_match_ud, kp2_match_ud)

ra=R+2.5
ta=t+2.5
rx=0.5-R
tx=2-t

mx= np.identity(3)
mi= np.array([1,2,1])
mi1=mi.transpose().reshape((3,1))

#Create the two rotational matrices
#Appending unit matrix and [0,0,0]
M_1_initial_camera = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
#Second camera will be at transformation [R|t]
M_2_second_camera_location = np.hstack((R, t))

#Tringulate the points
P_l = np.dot(left_intrinsics,  M_1_initial_camera)
P_r = np.dot(right_intrinsics,  M_2_second_camera_location)
point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(left_image_points, axis=1), np.expand_dims(right_image_points, axis=1))
point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_3d = point_4d[:3, :].T
point_3dx=point_3d

#getting the poses for camera
objp1 = np.float64([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
#change the name of file accordingly to image
cnr1= np.genfromtxt('../../parameters/left_1.csv',delimiter=',')
cnr2= np.genfromtxt('../../parameters/left_3.csv',delimiter=',')

assert cnr1.shape[0] == objp1.shape[0], 'points 3D and points 2D must have same number of vertices'
ret1, rvec1, tvec1 = cv2.solvePnP(objp1, cnr1, left_intrinsics, dist1)
ret2, rvec2, tvec2 = cv2.solvePnP(objp1, cnr2, left_intrinsics, dist1)

tvec1 = 5 * tvec1  # Scaling
tvec2 = 5 * tvec2  # Scaling

r1, _ = cv2.Rodrigues(rvec1) #rotation vector to rotation matrix
r2, _ = cv2.Rodrigues(rvec2) #rotation vector to rotation matrix

camera_position1 = -np.matrix(r1).T * np.matrix(tvec1)
camera_position2 = -np.matrix(r2).T * np.matrix(tvec2)

# plot with matplotlib
Ys = point_3dx[:,0]
Zs = point_3dx[:,1]
Xs = point_3dx[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Xs, Ys, Zs, c='b', marker='.')

plot_pyramid(ax, r1.T, camera_position1)
plot_pyramid(ax, r2.T, camera_position2)

ax.set_xlabel('Y')
ax.set_ylabel('Z')
ax.set_zlabel('X')
plt.title('3D point cloud')
plt.show()
plt.savefig('../../output/task_7/3dpoints.png')

cv2.waitKey(500)
cv2.destroyAllWindows()
