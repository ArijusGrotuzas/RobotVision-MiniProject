"""
Aruco marker detection and pose estimation
"""

import cv2 as cv
import cv2.aruco as ar
import numpy as np
import matplotlib.pyplot as plt


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
    pinkLower = (85, 125, 0)
    pinkUpper = (135, 255, 255)

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

    _, frame = cam.read()
    roi = cv.selectROI('', frame)
    r = roi

    while True:
        _, frame = cam.read()

        uvs = list()
        frame = frame[int(r[1]):int(r[1]+r[3]),
                      int(r[0]):int(r[0]+r[2])]

        # Convert to gray-scale to detect markers
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        bbox, ids, rejected = ar.detectMarkers(gray, arucoDict, parameters=arucoParams)

        # Convert back to rgb to draw markers
        ar.drawDetectedMarkers(frame, bbox, ids)

        # Detect a blue brick
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        binary_mask = cv.inRange(frame_HSV, pinkLower, pinkUpper)

        binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (4, 4)), iterations=2)
        binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (4, 4)), iterations=3)

        contours, _ = cv.findContours(binary_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Draw blue bricks center marker
        if contours is not None:
            for c in contours:
                moments = cv.moments(c)
                centroid = np.array([int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']), 1])
                uvs.append(centroid)
                cv.drawMarker(frame, (centroid[0], centroid[1]), (255, 0, 255), cv.MARKER_CROSS, 30)

        # If there are markers in the image, estimate the pose of the camera
        if type(ids) == np.ndarray:
            rotations, translations, _ = ar.estimatePoseSingleMarkers(bbox, 45, cameraMatrix, coefficients)

            # Find the marker with id 27 and re-project the 3D point
            index = np.where(ids == 1)
            if index[0].shape[0] != 0:
                transVec = translations[index][0]
                rotMat, jac = cv.Rodrigues(rotations[index][0])

                # Project center of aruco marker to image coordinate system
                _, scale = project3Dto2D(markerCenter, cameraMatrix, transVec, rotMat)

                if len(uvs) > 0:
                    projections = list()

                    for point in uvs:
                        # Project any given UV point to 3D
                        xyz = project2Dto3D(point, cameraMatrix, transVec, rotMat, scale)
                        projections.append(xyz)

                    print(projections)
                    plt.scatter(np.array(projections)[:, 0], np.array(projections)[:, 1])
                    plt.grid(True)
                    plt.show()

        cv.imshow('Output', frame)
        cv.waitKey(10)

if __name__ == '__main__':
    main()

