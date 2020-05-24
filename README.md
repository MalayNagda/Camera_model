# Camera_model
The project involves- 
1. [Stereo depth sensing](#1-stereo-depth-sensing) which is done in 4 steps-
   - Camera calibration by finding its intrinsic parameters.
   - Stereo calibration by calculating rotation and translation matrices between the two cameras in the stereo system. Stereo rectification by rotating the two views from the two cameras.
   - Sparse depth sensing by using epipolar geometry on pairs of feature points in the two views to triangulate the 3D points corresponding to those pairs. 
   - Dense depth sensing by calculating a disparity map of each pixel using the two views to obtain the depth of all pixels.
2. [Camera motion and structure](#2-camera-motion-and-structure) by calculating- 
   - homography to reconstruct the world plane from an image.
   - camera pose from a single view as a PnP problem.
   - camera pose from two views using essential matrix.

## 1. Stereo depth sensing

Depth sensing involves calculating the position of the 3D points in the world plane from 2D points in the image plane. This is done using two views of the same scene taken at the same time from two cameras which are part of a stereo system. The depth of only some of the matching pixels between the two views can be estimated using Sparse depth sensing. When the depth of all the pixels in the two views is estimated it is called dense depth estimation.   

## 2. Camera motion and structure