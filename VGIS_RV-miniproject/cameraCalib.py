""" Camera calibration """

import numpy as np
import cv2 as cv
import glob
import time


def projectionError(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix, coefficients):
    """
    Calculate the back-projection error, given intrinsic and extrinsic parameters.

    :param objectPoints: List
        A list of 3D points in the world coordinate system, for each image used for calibration.
    :param imagePoints: np.ndarray
        A list of 2D points in the image coordinate system i.e. projected points, for each image used for calibration.
    :param rvecs: np.ndarray
        A list of rotation vectors for each image used for calibration.
    :param tvecs: np.ndarray
        A list of translation vectors for each image used for calibration.
    :param cameraMatrix: np.ndarray
        A matrix of intrinsic camera parameters.
    :param coefficients: np.ndarray
        A list of distortion coefficients.
    """

    meanError = 0
    for i in range(len(objectPoints)):
        # Project 3D points to 2D points
        imagePoints_2D, _ = cv.projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, coefficients)

        # Get the error measurement using L2 normalization
        error = cv.norm(imagePoints[i], imagePoints_2D, cv.NORM_L2) / len(imagePoints_2D)
        meanError += error

    print(f'Mean back-projection error: {meanError / len(objectPoints)}, \n')


def getCalibration(size, directory='data/', display=False):
    """
    Get intrinsic parameters and distortion coefficients given some images and the size of chessboard squares

    :param size: Tuple
        Number of inner corners of the chessboard in x and y direction.
    :param directory: String
        Path to the images used for calibration.
    :param display: Boolean
        A flag parameter for displaying the detection of chessboard corners.

    :return: cameraMatrix, newCameraMatrix, coefficients: np.ndarray
        Camera matrix, camera matrix corrected by a free scale factor, and distortion coefficients.
    """

    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Initialize object points
    objectPoints = np.zeros((size[0] * size[1], 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:size[1], 0:size[0]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    newObjectPoints = list()  # 3d point in real world space
    imagePoints = list()  # 2d points in image plane.
    images = glob.glob(directory + '*.jpg')

    # Store the width and height of the image
    height, width = None, None

    # Traverse all images in the data folder
    for img in images:
        img = cv.imread(img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Get the size of the image
        height, width = gray.shape

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (size[1], size[0]), None)

        # If found, add object points, image points (after refining them)
        if ret:
            newObjectPoints.append(objectPoints)
            imagePoints.append(corners)

            if display:
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Draw and display the corners, Uncomment to see
                cv.drawChessboardCorners(img, (size[1], size[0]), corners2, ret)
                cv.imshow('img', cv.resize(img, (800, 500)))
                cv.waitKey(200)

    # Calibrate the camera, returns camera matrix and the distortion coefficients
    _, cameraMatrix, coefficients, rvecs, tvecs = cv.calibrateCamera(newObjectPoints, imagePoints, (width, height),
                                                                     None, None)

    # Correct the camera matrix
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, coefficients, (width, height), 1, (width, height))

    # Calculate the re-projection error
    projectionError(newObjectPoints, imagePoints, rvecs, tvecs, newCameraMatrix, coefficients)

    # Save the intrinsic parameters matrix and the distortion coefficients
    np.savez('intrinsic.npz', camera_matrix=cameraMatrix, distortion=coefficients)

    return cameraMatrix, newCameraMatrix, coefficients


def main():
    # Initialize parameters
    vid = cv.VideoCapture(0)
    ret, frame = vid.read()
    cameraMatrix, newCameraMatrix, distCoeffs = getCalibration((6,9))
    #dst = cv.undistort(frame, newCameraMatrix, distCoeffs)
    roi = cv.selectROI('',frame)
    # Capture the video frame
    # by frame
    #ret, frame = vid.read()
    cropped = frame[int(roi[1]):int(roi[1])+int(roi[3]),int(roi[0]):int(roi[0])+int(roi[2])]
    # Display the resulting frame
    cropped = cv.resize(cropped, (475, 308), interpolation = cv.INTER_AREA)
	
    cv.imwrite('CapturedImage.png',cropped)
    print(cropped.shape)
    time.sleep(.5)


if __name__ == '__main__':
    main()
