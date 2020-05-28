import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from cv2 import aruco

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import csv

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
left_intrinsics=left_intrinsics.astype('float64')
dist1=dist1.astype('float64')

translations = []
rotations = []
rotation_vec = []
positions = []

fig = plt.figure()
plt.axis('equal')
ax = fig.add_subplot(111, projection='3d')
for i in range(11):
    img = cv2.imread("../../images//task_6//left_" + str(i) + ".png")
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict)
    
    cnr = np.asarray(corners[0][0])
    csv_file_name = '../../parameters/left_'+str(i)  # getting the name of the file
    np.savetxt('{}.csv'.format(csv_file_name), cnr, delimiter=',', fmt='%f')
    
    img_marker = aruco.drawDetectedMarkers(img, corners, ids)
    cv2.imshow("Detecting Marker on image " + str(i), img_marker)
    cv2.waitKey(0)
    cv2.imwrite("../../output/task_6/Marker on image left_" + str(i) + ".png",img_marker)

	# img_corners = np.float32(corners).reshape(-1,2)
    objectpoints = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=np.float32)
	# img_points = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=np.float32)
    retval, rvec, tvec = cv2.solvePnP(objectpoints, corners[0][0], left_intrinsics, dist1)

    tvec = 5 * tvec #Scaling 

    translations.append(tvec)
    rotations.append(rvec)

	# Converting rotation vector to matrix
    rotM, a = cv2.Rodrigues(rvec)
    camera_position = -np.matrix(rotM).T * np.matrix(tvec)

    rotation_vec.append(rotM)

    xs = camera_position[0] # elements of np matrix
    ys = camera_position[1]
    zs = camera_position[2]

    ax.scatter(xs, ys, zs, c='b', marker='^', s = 0, zdir = 'z')
    ax.text(xs[0,0] + 0.01, ys[0,0] + 0.01, zs[0,0], 'left_'+str(i)+'.png', zdir='y')
    positions.append(camera_position)

    print("\n Image left_"+ str(i))
    print("R : ", rotM.T)
    print("\n T : ", camera_position)
    print("\n")


	## Code to draw pyramid
    vert = np.array([[-1, 1, 1, -1, 0], [-1, -1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    points = np.hstack((rotM.T, camera_position))
    points = np.append(points, [[0, 0, 0, 1]], axis=0)
	# print(points)
    vert_mul = (np.matrix(points) * np.matrix(vert))
    vert_mul = np.delete(vert_mul, 3, 0)
    vert_points = np.array(vert_mul.T)
    vertices = [[vert_points[0], vert_points[1], vert_points[4]], [vert_points[0], vert_points[3], vert_points[4]], [vert_points[2], vert_points[1], vert_points[4]], [vert_points[2], vert_points[3], vert_points[4]], [vert_points[0], vert_points[1], vert_points[2], vert_points[3]]]
	# print(vertices)
    ax.add_collection3d(Poly3DCollection(vertices, facecolors='white', linewidths=1, edgecolors='k', alpha=.25))


## Plot for square with red dot at the corner in 3D
square_corners = np.array([[0, 5, 0], [5, 5, 0], [5, 0, 0], [0, 0, 0]])
square_verts = [[square_corners[0], square_corners[1], square_corners[2], square_corners[3]]]
m_points = np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]])
    # ax.scatter3D(m_pts[:, 0], m_pts[:, 1], m_pts[:, 2])
red_corner = [[m_points[0], m_points[1], m_points[2], m_points[3]]]

    # # plot sides
ax.add_collection3d(Poly3DCollection(square_verts,
                                         facecolors='black', linewidths=1, edgecolors='k', alpha=.5))

ax.add_collection3d(Poly3DCollection(red_corner,
                                         facecolors='red', linewidths=1, edgecolors='r', alpha=1))
ax.set_xlim(-15, 15)
ax.set_ylim(10,-10)
ax.set_zlim(20,-20)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
plt.savefig('../../output/task_6/camera_pose.png',dpi=600, bbox_inches = "tight")
