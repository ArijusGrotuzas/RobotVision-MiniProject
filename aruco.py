"""
Aruco marker detection and pose estimation
"""

import cv2 as cv
import cv2.aruco as ar
import numpy as np


def project3Dto2D(point, cameraMatrix, transVec, rotMat):
    """
    Project a (x, y, z, 1) point in an arbitrary coordinate system e.g marker, world.

    :param point: np.ndarray
        A (x, y, z, 1) point in world coordinate system
    :param cameraMatrix: np.ndarray
        A (3 x 3) matrix that contains the intrinsic camera parameters.
    :param transVec: np.ndarray
        A 3D translational vector.
    :param rotMat:
        A (3 x 3) rotation matrix.

    :return: np.ndarray, Integer
        A (u, v) coordinate of the image and a scaling factor
    """

    rotTransMat = np.column_stack((rotMat, transVec))
    projectionMat = cameraMatrix.dot(rotTransMat)
    suv = projectionMat.dot(point.T)
    s = suv[2]
    return suv / s, s


def project2Dto3D(uv, cameraMatrix, transVec, rotMat, scaleFactor):
    """
    Project a (u, v) coordinate of the image to world coordinate system given extrinsic and intrinsic parameters.

    :param uv: np.ndarray
        A (u, v) coordinate of the image.
    :param cameraMatrix: np.ndarray
        A matrix that contains the intrinsic camera parameters.
    :param transVec: np.ndarray
        A 3D translational vector.
    :param rotMat:
        A (3 x 3) rotation matrix.
    :param scaleFactor: Float
        Scaling factor.

    :return: np.ndarray
        A (x, y, z) point in world coordinate system
    """

    suv = scaleFactor * uv
    invCamMat = np.linalg.inv(cameraMatrix)
    xyzCam = invCamMat.dot(suv)
    xyzCam = xyzCam - transVec
    invRotMat = np.linalg.inv(rotMat)
    return np.round(invRotMat.dot(xyzCam), 1)


def main():
    # Load the aruco marker dictionary
    arucoDict = ar.Dictionary_get(ar.DICT_4X4_50)
    arucoParams = ar.DetectorParameters_create()

    # Open the camera stream
    cam = cv.VideoCapture(0 + cv.CAP_DSHOW)

    # Load the intrinsic parameters
    data = np.load('intrinsic.npz')
    cameraMatrix = data['camera_matrix']
    coefficients = data['distortion']

    # Define center point of the marker in marker's coordinate system
    markerCenter = np.array([0.0, 0.0, 0.0, 1.0])

    while True:
        _, frame = cam.read()

        # Convert to gray-scale to detect markers
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        bbox, ids, rejected = ar.detectMarkers(gray, arucoDict, parameters=arucoParams)

        # Convert back to rgb to draw markers
        result = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        ar.drawDetectedMarkers(result, bbox, ids)

        # If there are markers in the image, estimate the pose of the camera
        if type(ids) == np.ndarray:
            rotations, translations, _ = ar.estimatePoseSingleMarkers(bbox, 4, cameraMatrix, coefficients)

            # Find the marker with id 27 and re-project the 3D point
            index = np.where(ids == 27)
            if index[0].shape[0] != 0:
                transVec = translations[index][0]
                rotMat, jac = cv.Rodrigues(rotations[index][0])

                # Project center of aruco marker to image coordinate system
                uv, scale = project3Dto2D(markerCenter, cameraMatrix, transVec, rotMat)
                cv.drawMarker(result, (int(uv[0]), int(uv[1])), (255, 255, 0), markerType=cv.MARKER_CROSS)

                # Project any given UV point to 3D
                xyz = project2Dto3D(uv, cameraMatrix, transVec, rotMat, scale)
                print(xyz)

            # Draw orientation axes, Uncomment to see
            # for idx, rot in enumerate(rotations):
            #     trans = translations[idx]
            #     cv.drawFrameAxes(result, cameraMatrix, coefficients, rot, trans, 0.05)

        cv.imshow('Output', result)
        cv.waitKey(10)


if __name__ == '__main__':
    main()
