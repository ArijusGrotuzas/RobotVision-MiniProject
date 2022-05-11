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
#RDK.setRunMode(RUNMODE_RUN_ROBOT) #uncomment for running on robot
SIZE_BOX_Z = 50

#Calc Homography matrix from previously extracted red locations
# to calculate the transformation matrix
input_pts = np.float32([[31, 252],[52, 47],[439, 38],[424, 262]])
output_pts = np.float32([[-446.7, -368.5],[-296.7, -225.1],[-34.9, -450.6],[-196.8, -603.5]])


# Compute the perspective transform M
H = np.float32(cv2.getPerspectiveTransform(input_pts,output_pts))
print('Homography Matrix: \n', H)


#Image Processing stuff
class ShapeDetector:
    def __init__(self):
        pass
    def detect(self, c):
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    
        if len(approx) == 3:
            ((x, y), (w, h), r) = cv2.minAreaRect(approx)
            ar = w / float(h)
            shape = "triangle"
        elif len(approx) == 4:
            ((x, y), (w, h), r) = cv2.minAreaRect(approx)
            ar = w / float(h)
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "circle"
        return shape, r 

class ColorLabeler:
    def __init__(self):
        colors = OrderedDict({
            "orange": (255, 95, 0),
            "green": (0, 155, 130),
            "yellow": (210, 170, 0),
            "black": (50, 50, 50),
            "red": (255, 0, 0),
            "blue": (0, 90, 150)})
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []
        for (i, (name, rgb)) in enumerate(colors.items()):
            self.lab[i] = rgb
            self.colorNames.append(name)
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)
        
    def label(self, image, c):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]
        minDist = (np.inf, None)
        for (i, row) in enumerate(self.lab):
            d = dist.euclidean(row[0], mean)
            if d < minDist[0]:
                minDist = (d, i)
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
        LUT = cv2.LUT(image, table)
        return LUT
    gamma = adjust_gamma(image, gamma=2.6)
    blur = cv2.GaussianBlur(gamma,(5,5),cv2.BORDER_REFLECT)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 10
    attempts=10
    twoDimage = blur.reshape((-1,3))
    twoDimage = np.float32(twoDimage)
    ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    K_image = res.reshape((image.shape))

    lab = cv2.cvtColor(K_image, cv2.COLOR_BGR2LAB)
    img_hsv=cv2.cvtColor(K_image, cv2.COLOR_BGR2HSV)
    img_gray=cv2.cvtColor(img_hsv, cv2.COLOR_BGR2GRAY)

    ret, img_tresh = cv2.threshold(img_gray, 90, 255,cv2.THRESH_BINARY)
    ret, B_img_tresh = cv2.threshold(img_gray, 10, 255,cv2.THRESH_BINARY)
    B_img_tresh = cv2.bitwise_not(B_img_tresh)
    img_tresh = img_tresh + B_img_tresh
    arr_cnt = cv2.findContours(img_tresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(arr_cnt)
    sd = ShapeDetector()
    cl = ColorLabeler()
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        shape, rotation = sd.detect(c)
        color = cl.label(lab, c)

        c = c.astype("float")
        c = c.astype("int")

        #Transofrm pixel coordinates to world coordinates
        world_coords = np.squeeze(cv2.perspectiveTransform(np.float32([cX,cY]).reshape(-1,1,2).astype(np.float32), H), axis=1)[0]
        text = "{} {} {} {}deg".format(color, int(world_coords[0]), int(world_coords[1]), int(rotation))
        print(color, shape,cX,cY)
        print('table coodinates: ', world_coords[0], world_coords[1])

        #insert brick woth pose and color into list to be used later
        lego_brick_list.append([world_coords[0],world_coords[1],color,math.radians(rotation)])
        
    return lego_brick_list
                            
            
def WaitPartCamera():
    #Use real image processing and brick detection
    print("Loading image:" + 'CapturedImage.png')   
    
    # Implement your image processing here:
    tt = cv2.imread('CapturedImage.png')
    return detect_bricks('/Users/maltheesbensen/Desktop/VGIS_RV-miniproject/CapturedImage.png')


#Robot stuff
#
#

def TCP_On(toolitem):
    """Close the gripper"""
    robot.setDO(0, 0)
    robot.setDO(1, 1)
    time.sleep(0.5)
        
def TCP_Off(toolitem, itemleave=0):
    """Open the gripper"""
    robot.setDO(1,0)
    robot.setDO(0, 1)
    time.sleep(0.5)

def find_brick(brick_list, color):
    """ Find brick of desired color in list an remove element
    """
    for i in range(len(brick_list)):
        if(brick_list[i][2] == color):
            x=brick_list[i][0]
            y=brick_list[i][1]
            r=brick_list[i][3]
            #remove brick from list to not use it again
            brick_list.pop(i)
            
            return x,y,r

def build_figure(figure_frame, colors):
    poseref = figure_frame.Pose()
    for i in range(len(colors)):
        x,y,r = find_brick(lego_brick_list,colors[i])
        robot.MoveL(Pose(x,y,SIZE_BOX_Z*4,180,0,0)*rotz(r)*rotz(math.radians(45)))
        robot.WaitMove()
        robot.MoveL(Pose(x,y,SIZE_BOX_Z+38,180,0,0)*rotz(r)*rotz(math.radians(45)))
        robot.WaitMove()
        time.sleep(0.2)
        TCP_On(tool)
        robot.WaitMove()
        robot.MoveL(Pose(x,y,SIZE_BOX_Z*4,180,0,0)*rotz(r)*rotz(math.radians(45)))
        robot.WaitMove()
        posei = poseref*transl(0,0,-SIZE_BOX_Z*3) #calc new pose of brick for figure
        robot.MoveL(posei)
        time.sleep(0.2)
        poseij = poseref*transl(0,0,-SIZE_BOX_Z/2.5*i+5) #calc new pose of brick for figure
        robot.MoveL(poseij)
        time.sleep(0.2)
        TCP_Off(tool)
        posei = poseref*transl(0,0,-SIZE_BOX_Z*3) #calc new pose of brick for figure
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
    print("Status: {0}, status msg: {1}".format(status, status_msg))
    robot.setDO(1,0)
    robot.setDO(0, 1)
time.sleep(2)

robot.setSpeed(speed_linear=150, speed_joints=30)
robot.setPoseTool(tool)
        
lego_brick_list = WaitPartCamera()

# get the target frames:
home = RDK.Item('Home')
homer = RDK.Item('Homer')
marge = RDK.Item('marge')
lisa = RDK.Item('Lisa')
bart = RDK.Item('Bart')
maggie = RDK.Item('Maggie')


#Move Robot Home
robot.MoveL(home)

homer_colors = ['blue',  'black', 'yellow']
maggie_colors = ['blue', 'yellow']
bart_colors = ['blue',  'orange', 'yellow']
lisa_colors = ['yellow',  'orange', 'yellow']
marge_colors = ['green', 'yellow', 'blue']

#Build Homer
build_figure(homer, homer_colors)
build_figure(maggie, maggie_colors)
build_figure(lisa, lisa_colors)
build_figure(bart, bart_colors)
build_figure(marge, marge_colors)

#Move Home after finish
robot.MoveL(home)
