import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
 
mtx = np.array([[423.27381306, 0, 341.34626532], [0, 421.27401756, 269.28542111], [0, 0, 1]])
dist = np.array([-0.43394157423038077, 0.26707717557547866, -0.00031144347020293427, 0.0005638938101488364, -0.10970452266148858])

mtx_right = np.array([[420.91160482, 0, 352.16135589], [0, 418.72245958, 264.50726699], [0, 0, 1]])
dist_right = np.array([-0.4145817681176909, 0.19961273246897668, -0.00014832091141656534, -0.0013686760437966467, -0.05113584625015141])

world_points = np.array([[300, 800,   0],
 [310, 800,   0],
 [320, 800,   0],
 [330, 800,   0],
 [340, 800,   0],
 [350, 800,   0],
 [360, 800,   0],
 [370, 800,   0],
 [380, 800,   0],
 [300, 810,   0],
 [310, 810,   0],
 [320, 810,   0],
 [330, 810,   0],
 [340, 810,   0],
 [350, 810,   0],
 [360, 810,   0],
 [370, 810,   0],
 [380, 810,   0],
 [300, 820,   0],
 [310, 820,   0],
 [320, 820,   0],
 [330, 820,   0],
 [340, 820,   0],
 [350, 820,   0],
 [360, 820,   0],
 [370, 820,   0],
 [380, 820,   0],
 [300, 830,   0],
 [310, 830,   0],
 [320, 830,   0],
 [330, 830,   0],
 [340, 830,   0],
 [350, 830,   0],
 [360, 830,   0],
 [370, 830,   0],
 [380, 830,   0],
 [300, 840,   0],
 [310, 840,   0],
 [320, 840,   0],
 [330, 840,   0],
 [340, 840,   0],
 [350, 840,   0],
 [360, 840,   0],
 [370, 840,   0],
 [380, 840,   0],
 [300, 850,   0],
 [310, 850,   0],
 [320, 850,   0],
 [330, 850,   0],
 [340, 850,   0],
 [350, 850,   0],
 [360, 850,   0],
 [370, 850,   0],
 [380, 850,   0]], dtype=np.float32)

## Undistorting left_0 image 
img_1 = cv2.imread("../../images//task_5//left_0.png")
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, None,(640,480),cv2.CV_32FC1)
undistored_image_1 = cv2.remap(img_1, mapx, mapy, cv2.INTER_LINEAR)
cv2.imshow("Undistorting left_0 image", undistored_image_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Finding corners of the chess borad in left_0 image
gray_1 = cv2.cvtColor(undistored_image_1, cv2.COLOR_BGR2GRAY)
ret, corners_1 = cv2.findChessboardCorners(gray_1, (9,6),None)
corners_1 = np.float32(corners_1).reshape(-1,2)

## Drawing corners on to image to verify
img_corner_1 = cv2.drawChessboardCorners(undistored_image_1, (9,6), corners_1, ret)
cv2.imshow("Corners Detected on left_0", img_corner_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Undistorting left_1 image
img_2 = cv2.imread("../../images//task_5//right_0.png")
mapx_2, mapy_2 = cv2.initUndistortRectifyMap(mtx, dist, None, None,(640,480),cv2.CV_32FC1)
undistored_image_2 = cv2.remap(img_2, mapx_2, mapy_2, cv2.INTER_LINEAR)
cv2.imshow("Undistorting right_0 image", undistored_image_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Finding corners of the chess borad in left_0 image
gray_2 = cv2.cvtColor(undistored_image_2, cv2.COLOR_BGR2GRAY)
ret, corners_2 = cv2.findChessboardCorners(gray_2, (9,6),None)
corners_2 = np.float32(corners_2).reshape(-1,2)

## Drawing corners on to image to verify
img_corner_2 = cv2.drawChessboardCorners(undistored_image_2, (9,6), corners_2, ret)
cv2.imshow("Corners Detected on right_0", img_corner_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

M_1, mask = cv2.findHomography(corners_1, world_points, cv2.RANSAC, 5.0)
M_2, mask = cv2.findHomography(corners_2, world_points, cv2.RANSAC, 5.0)
print('Homography matrix for image 1\n',M_1)
print('Homography matrix for image 2\n',M_2)

img_perspective_1 = cv2.warpPerspective(img_corner_1, M_1, (960, 1280))
img_perspective_2 = cv2.warpPerspective(img_corner_2, M_2, (960, 1280))
cv2.imshow("perspective", cv2.resize(img_perspective_1, (img_1.shape[0], img_1.shape[1]), interpolation = cv2.INTER_AREA))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("perspective", cv2.resize(img_perspective_2, (img_1.shape[0], img_1.shape[1]), interpolation = cv2.INTER_AREA))
cv2.waitKey(0)
cv2.destroyAllWindows()

fig, (ax1,ax2) = plt.subplots(1, 2)
ax1.imshow(cv2.cvtColor(img_perspective_1 , cv2.COLOR_BGR2RGB), interpolation=None)
ax2.imshow(cv2.cvtColor(img_perspective_2 , cv2.COLOR_BGR2RGB),interpolation=None)
# ax3.imshow(undistored_image2,interpolation=None)
ax2.set_xlabel('\n 2D world reconstruction from the left_0 and right_0 image')
plt.show()
plt.savefig('../../output/task_5/twoD_recontruct.png',dpi=600, bbox_inches = "tight")
