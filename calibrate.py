# Import required modules
import cv2
import numpy as np
import os
import glob

def calibrate(directory_glob):
    # Define the dimensions of checkerboard
    CHECKERBOARD = (5, 7)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Vector for 3D points
    threedpoints = []
    # Vector for 2D points
    twodpoints = []

    # 3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory. If no path specified, takes full directory
    # images = glob.glob('./data/*.png')
    images = glob.glob(directory_glob)

    count = 0
    for filename in images:
        image = cv2.imread(filename)
        height, width = image.shape[:2]
        image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        count += 1

        # # DEBUG
        # #output the current image
        # plt.imshow(image)
        # plt.title("Image " + str(count))
        # plt.axis('off')
        # plt.show()

    # Find the chess board corners
    # If desired number of corners are
    # found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
                        grayColor, CHECKERBOARD,
                        cv2.CALIB_CB_ADAPTIVE_THRESH
                        + cv2.CALIB_CB_FAST_CHECK +
                        cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If desired number of corners can be detected then,
    # refine the pixel coordinates and display
    # them on the images of checker board
        if ret == True:
            threedpoints.append(objectp3d)

            # Refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
            twodpoints.append(corners2)
            
            # DEBUG
        #     # Draw and display the corners
        #     image = cv2.drawChessboardCorners(image,
        #         CHECKERBOARD,
        #         corners2, ret)

        # plt.imshow(image)
        # plt.title("Corners detected")
        # plt.axis('off')
        # plt.show()

    h, w = image.shape[:2]

    # Calibrate by passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints)

    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)

    # DEBUG
    # # Values
    # print(" Camera matrix:")
    # print(matrix)
    # print("\n Distortion coefficient:")
    # print(distortion)
    # print("\n Rotation Vectors:")
    # print(r_vecs)
    # print("\n Translation Vectors:")
    # print(t_vecs)

    return (matrix, distortion)

if __name__ == "__main__":
    # change to whatever folder your checkerboard images are in
    k, distortion_params = calibrate('/data/checkerboard/*.png')
