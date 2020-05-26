import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
from mpl_toolkits.mplot3d import Axes3D

fx = 423.27381306
fy = 421.27401756
cx = 341.34626532
cy = 269.28542111

K = np.float64([[fx, 0, cx], 
                [0, fy, cy], 
                [0, 0, 1]])

D = np.float64([-0.43394157423038077, 0.26707717557547866, -0.00031144347020293427, 0.0005638938101488364,-0.10970452266148858])

img1 = cv2.imread('../../images/task_7/left_4.png',0)

img2 = cv2.imread('../../images/task_7/left_5.png',0)

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objpoints = []
imagepoints = []
objpoints.append(objp)
objpoints = np.float64(objpoints)

T = np.float64([[-2.45477982343564],
                [-0.0332453385522794],
                [-0.0198820833667838]])

T = np.transpose(T)

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
#T=np.reshape([params[5],params[6],params[7]],(3,1))
E=np.reshape([params[9],params[10],params[11]],(3,3))
F=np.reshape([params[13],params[14],params[15]],(3,3))
R=R.astype('float64')
#T=T.astype('float64')
E=E.astype('float64')
F=F.astype('float64')

h,  w = img1.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,D,(w,h),1,(w,h))
mapx,mapy = cv2.initUndistortRectifyMap(K,D,None,None,(w,h),cv2.CV_32FC1)
img1 = cv2.remap(img1,mapx,mapy,cv2.INTER_LINEAR)

h2,  w2 = img2.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,D,(w2,h2),1,(w2,h2))
mapx2,mapy2 = cv2.initUndistortRectifyMap(K,D,None,None,(w2,h2),cv2.CV_32FC1)
img2 = cv2.remap(img2,mapx2,mapy2,cv2.INTER_LINEAR)


for i in range(len(objpoints)):
    imgpts, _ = cv2.projectPoints(objpoints[i], R[i], T[i], K, D)
    imagepoints.append(imgpts)

#print(imagepoints)

gr1 = img1
gr2= img2
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:200],None, flags=2)

plt.imshow(img3)
plt.savefig('../../output/task_7/Figure3.png',dpi=600, bbox_inches = "tight")
plt.show()




kp1_match = np.array([kp1[mat.queryIdx].pt for mat in matches])
kp2_match = np.array([kp2[mat.trainIdx].pt for mat in matches])

kp1_match_ud = cv2.undistortPoints(np.expand_dims(kp1_match,axis=1),K,D)
kp2_match_ud = cv2.undistortPoints(np.expand_dims(kp2_match,axis=1),K,D)

E, mask_e = cv2.findEssentialMat(kp1_match_ud, kp2_match_ud, focal=1.0, pp=(0., 0.), 
                               method=cv2.RANSAC, prob=0.999, threshold=0.001)
print("The calculated essential matrix is",E)
#print ("Essential matrix: used ",np.sum(mask_e) ," of total ",len(matches),"matches")

points, R, t, mask_RP = cv2.recoverPose(E, kp1_match_ud, kp2_match_ud, mask=mask_e)
#print("points:",points,"\trecover pose mask:",np.sum(mask_RP!=0))
print("R:",R,"t:",t.T)

bool_mask = mask_RP.astype(bool)
img_valid = cv2.drawMatches(img1,kp1,img2,kp2,matches, None, 
                            matchColor=(0, 255, 0), 
                            matchesMask=bool_mask.ravel().tolist(), flags=2)

plt.imshow(img_valid)
plt.savefig('../../output/task_7/Figure4.png',dpi=600, bbox_inches = "tight")
plt.show()

