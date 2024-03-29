# Robot Vision Mini-Project

The task for the Robot Vision Mini-Project was to automate an assembly of Simpson figures made out of lego Duplo bricks. The robot cell came equipped with a camera, robot manipulator (`UR 5`), and an assembly station.

In order for the project to work following steps must be executed in a following order:

1. Take picture using `cameraCalib.py`;
2. Find homography matrix using `Detect_lego_and_calibrate_points.ipynb`;
3. `VGIS_RV-miniproject.rdk` is ready to run;
4. Uncomment line 12 in `create_figures.py` within the RoboDK environment and change line 235 to robot IP.


# Table Of Contents

- [Camera Calibration](#Camera-Calibration)
- [Robot Simulation](#Robot-Simulation)
- [Physical Setup](#Physical-Setup)
- [Computer Vision Pipeline](#Computer-Vision-Pipeline)

# Camera Calibration

The camera calibration was performed to find the intrinsic camera parameters. These parameters would then be used for converting the `2D` coordinates in the image to the `3D` coordinates relative to the base of the robot.

|  |  |
| :---:| :---:|
| ![cam (2)](https://user-images.githubusercontent.com/50104866/182015716-0f02259d-1962-42db-8879-939ec5001a10.jpg) | ![cam (5)](https://user-images.githubusercontent.com/50104866/182015720-f90b5440-28e3-41c9-bc8e-a9d1f4de7019.jpg)|
| ![cam (6)](https://user-images.githubusercontent.com/50104866/182015741-8125ac25-7d7b-448e-9c00-a519afd0b153.jpg)| ![cam (9)](https://user-images.githubusercontent.com/50104866/182015746-737c7a7e-d2f3-4bc4-8388-88787da97919.jpg) |


# Robot Simulation

The simulation of a robot cell was achieved using the `RoboDK` environment, to program the robot manipulator offline. The environment was connected via an ethernet cable to the manipulator. The simulation created in `RoboDK` was then directly transferred and executed in the physical robot cell.  

<img width="869" alt="robodk" src="https://user-images.githubusercontent.com/50104866/182015703-a9fe0a7a-fc11-40a8-a6ea-7e731bf8d200.png">


# Physical Setup

The physical setup of the robot cell consisted of a camera, robot manipulator, workspace, and a laptop connected to the robot via ethernet.

![Physical_setup](https://user-images.githubusercontent.com/50104866/182016274-65112dd1-3c9f-4a0a-9a33-020812588337.png)

# Computer Vision Pipeline

To classify a set of Lego bricks based on their color, and to derive their position in the image, multiple Computer Vision algorithms were carried out in the following order:

### Initial Image
![CapturedImage](https://user-images.githubusercontent.com/45823340/167778541-cf4b2391-7d58-4c3b-b12b-fb68d525dc25.png)

### Gamma Mapping
![1_gamma_mapped](https://user-images.githubusercontent.com/45823340/167778552-028b8fa1-a244-41db-a1d5-2d6cca51fded.png)

### Gaussian Blurring
![2_blurred](https://user-images.githubusercontent.com/45823340/167778575-ea3217f4-5082-4e28-b461-bcdd970d74ae.png)

### K-means Clustering
![3_kmean](https://user-images.githubusercontent.com/45823340/167778596-42892e0c-b97c-43b8-91a1-18b65c7ee19d.png)

### Colorspace Conversion and Binary Thresholding
![4_colorspace_conv_and_thresh](https://user-images.githubusercontent.com/45823340/167808436-e0196516-f57b-4539-9e85-fb2edea0573a.png)

### Mapped Bricks
![5_detected](https://user-images.githubusercontent.com/45823340/167778628-fb9892af-eb2b-4fb8-a622-84df85d01100.png)

