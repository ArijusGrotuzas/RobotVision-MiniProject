from robodk.robolink import *      # RoboDK's API
from robodk.robomath import *      # Math toolbox for robots
import cv2
import imutils
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import time

RDK = Robolink()
#RDK.setRunMode(RUNMODE_RUN_ROBOT) uncomment for running on robot
SIZE_BOX_Z = 50

#Calc Homography matrix from previously extracted red locations
# to calculate the transformation matrix
input_pts = np.float32([[36, 264],[443, 273],[445, 20],[49, 29]])
output_pts = np.float32([[-449.8, -344.0],[-161.5, -628.7],[14, -442.5],[-283.0, -183.0]])


# Compute the perspective transform M
H = np.float32(cv2.getPerspectiveTransform(input_pts,output_pts))
print('Homography Matrix: \n', H)


#Image Processing stuff
class ShapeDetector:
    def __init__(self):
        pass
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            ((x, y), (w, h), r) = cv2.minAreaRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape, r #convert to deg fopr visu

class ColorLabeler:
    def __init__(self):
        # initialize the colors dictionary, containing the color
        # name as the key and the RGB tuple as the value
        colors = OrderedDict({
            "orange": (255, 0.1*255, 0),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "blue": (0, 0, 255)})
        # allocate memory for the L*a*b* image, then initialize
        # the color names list
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []
        # loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)
        # convert the L*a*b* array from the RGB color space
        # to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)
        
    def label(self, image, c):
        # construct a mask for the contour, then compute the
        # average L*a*b* value for the masked region
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]
        # initialize the minimum distance found thus far
        minDist = (np.inf, None)
        # loop over the known L*a*b* color values
        for (i, row) in enumerate(self.lab):
            # compute the distance between the current L*a*b*
            # color value and the mean of the image
            d = dist.euclidean(row[0], mean)
            # if the distance is smaller than the current distance,
            # then update the bookkeeping variable
            if d < minDist[0]:
                minDist = (d, i)
        # return the name of the color with the smallest distance
        return self.colorNames[minDist[1]]


def detect_bricks(path):
    lego_brick_list = []
    # load the image
    image = cv2.imread(path)
    def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    resized = adjust_gamma(resized, gamma=2.6)
    blur = cv2.GaussianBlur(resized,(5,5),cv2.BORDER_REFLECT)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    img_hsv=cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    img_gray=cv2.cvtColor(img_hsv, cv2.COLOR_BGR2GRAY)
    ret, img_tresh = cv2.threshold(img_gray, 100, 255,cv2.THRESH_BINARY)

    arr_cnt = cv2.findContours(img_tresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(arr_cnt)
    sd = ShapeDetector()
    cl = ColorLabeler()

    # loop over the contours
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape, rotation = sd.detect(c)
        color = cl.label(lab, c)
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        print(color,shape,cX,cY)

        #Transofrm pixel coordinates to world coordinates
        world_coords = np.squeeze(cv2.perspectiveTransform(np.float32([cX,cY]).reshape(-1,1,2).astype(np.float32), H), axis=1)[0]
        text = "{} {} {} {}deg".format(color, int(world_coords[0]), int(world_coords[1]), int(rotation))
        print(color, shape,cX,cY)
        print('table coodinates: ', world_coords[0], world_coords[1])

        #insert brick woth pose and color into list to be used later
        lego_brick_list.append([world_coords[0],world_coords[1],math.radians(rotation),color,rotation])
        
    return lego_brick_list
                            

def cleanup(objects, startswith="lego_brick"):
    """Deletes all objects where the name starts with "startswith", from the provided list of objects."""    
    for item in objects:
        if item.Name().startswith(startswith):
            item.Delete()
            
def WaitPartCamera():
    #Use real image processing and brick detection
    print("Saving camera snapshot to file:" + 'Image-Camera-Simulation.png')   
    
    # Implement your image processing here:
    return detect_bricks('Image-Camera-Simulation.png')


#Robot stuff
#
#

def TCP_On(toolitem):
    """Close the gripper"""
    robot.setDO(0, 0)
    robot.setDO(1, 1)
    time.sleep(0.2)
        
def TCP_Off(toolitem, itemleave=0):
    """Open the gripper"""
    robot.setDO(1,0)
    robot.setDO(0, 1)
    time.sleep(0.2)

def find_brick(brick_list, color):
    """ Find brick of desired color in list an remove element
    """
    for i in range(len(brick_list)):
        if(brick_list[i][3] == color):
            x=brick_list[i][0]
            y=brick_list[i][1]
            r=brick_list[i][2]
            #remove brick from list to not use it again
            brick_list.pop(i)
            
            return x,y,r

def build_figure(figure_frame, colors):
    poseref = figure_frame.Pose()
    for i in range(len(colors)):
        x,y,r = find_brick(lego_brick_list,colors[i])
        robot.MoveL(Pose(x,y,SIZE_BOX_Z*4,180,0,0)*rotz(r))
        robot.WaitMove()
        robot.MoveL(Pose(x,y,SIZE_BOX_Z,180,0,0)*rotz(r))
        robot.WaitMove()
        TCP_On(tool)
        robot.WaitMove()
        robot.MoveL(Pose(x,y,SIZE_BOX_Z*4,180,0,0)*rotz(r))
        robot.WaitMove()
        posei = poseref*transl(0,0,-SIZE_BOX_Z*4) #calc new pose of brick for figure
        robot.MoveL(posei)
        time.sleep(0.2)
        poseij = poseref*transl(0,0,-SIZE_BOX_Z/2.5*i) #calc new pose of brick for figure
        robot.MoveL(poseij)
        time.sleep(0.2)
        TCP_Off(tool)
        posei = poseref*transl(0,0,-SIZE_BOX_Z*4) #calc new pose of brick for figure
        robot.MoveL(posei)
        time.sleep(0.2)
        posei = poseij

    
   

# gather robot, tool and reference frames from the station
robot               = RDK.Item('UR5', ITEM_TYPE_ROBOT)
tool                = RDK.Item('Gripper', ITEM_TYPE_TOOL)
table               = RDK.Item('World', ITEM_TYPE_FRAME)
if RDK.RunMode() == RUNMODE_RUN_ROBOT:
    robot.setConnectionParams('192.168.87.110',30000,'/', 'anonymous','')
    robot.ConnectSafe()
    status, status_msg = robot.ConnectedState()
    print("Status: {0}, status msg: {1}".format(status, status_msg)) # This returned "0 Ready"
    robot.setDO(1,0)
    robot.setDO(0, 1)
time.sleep(2)

robot.setSpeed(speed_linear=110, speed_joints=30)
robot.setPoseTool(tool)
        
lego_brick_list = WaitPartCamera()
print(lego_brick_list)

# get the home target frames:
home = RDK.Item('Home')
homer = RDK.Item('Homer')
marge = RDK.Item('marge')
lisa = RDK.Item('Lisa')
bart = RDK.Item('Bart')
maggie = RDK.Item('Maggie')


#Move Robot Home and get Image from table
robot.MoveL(home)

homer_colors = ['blue',  'blue', 'yellow']
maggie_colors = ['blue', 'yellow']
bart_colors = ['blue',  'orange', 'yellow']
lisa_colors = ['yellow',  'orange', 'yellow']
marge_colors = ['orange', 'yellow', 'blue']

#Build Homer
build_figure(homer, homer_colors)
build_figure(maggie, maggie_colors)
build_figure(lisa, lisa_colors)
build_figure(bart, bart_colors)
build_figure(marge, marge_colors)

#Move Home after finish
robot.MoveL(home)
