import cv2
import numpy as np
import time


#color arrays
MeanColor = {
    "blue": [100,250],
    "red": [0,250],
    "yellow": [30,250],
    "green": [70,250],
    "white": [0,0]
}

# character arrays
character_map = {
    "homer": ['blue','white','yellow'],
    "bart": ['blue','red','yellow'],
    "marge": ['green','yellow','blue'],
    "lisa" : ['yellow','red','yellow'],
    "maggie" : ['blue','yellow']
}


# List for holding brick colors
colors = []


def get_lego_centroids(imagepath,backgroundpath):
    img_example=cv2.imread(imagepath)
    img_bg=cv2.imread(backgroundpath)
    img_bg_hsv=cv2.cvtColor(img_bg, cv2.COLOR_BGR2HSV)
    img_hsv=cv2.cvtColor(img_example, cv2.COLOR_BGR2HSV)
    img_bg_gray=cv2.cvtColor(img_bg_hsv, cv2.COLOR_BGR2GRAY)
    img_gray=cv2.cvtColor(img_hsv, cv2.COLOR_BGR2GRAY)
    diff_gray=cv2.absdiff(img_bg_gray,img_gray)
    diff_gray_blur = cv2.GaussianBlur(diff_gray,(5,5),cv2.BORDER_REFLECT)
    ret, img_tresh = cv2.threshold(diff_gray_blur, 38, 255,cv2.THRESH_BINARY)
    arr_cnt, a2 = cv2.findContours(img_tresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img_with_allcontours=img_example.copy()
    cv2.drawContours(img_with_allcontours, arr_cnt, -1, (0,255,0), 3)
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_tresh, 8, cv2.CV_32S)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 150
    cents = np.zeros((centroids.shape))
    img2 = np.zeros((output.shape))
    cent_c = 0
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            cents[cent_c] = centroids[i+1]
            cent_c+=1
            img2[output == i + 1] = 255
    cents= cents[:cent_c]
    
    cent_cols=np.zeros((cents.shape[0],3))
    count = 0
    for i in cents:
        cent_cols[count] = img_hsv[int(i[1])][int(i[0])]
        count+=1
    return cents,cent_cols

def retCol(h, s, cords):
    hsv = cv2.merge((h,s))
    for col in MeanColor:
        lower = np.array(MeanColor[col])-10
        upper = np.array(MeanColor[col])+10
        mask = cv2.inRange(hsv,lower,upper)
        if 255 in mask:
            colors.insert(cords,col)
            break

def getlegocolors(cents,cent_cols):
    for i in range(len(cents)):
        retCol(cent_cols[i][0],cent_cols[i][1],i)
    return colors
    
    
    
#run returns the array of pixel coordinates of lego bricks needed to build a character
# It takes character name and path of background image and image with all the lego bricks
def run(character,imagepath,backgroundpath):
    cents,cent_cols = get_lego_centroids(imagepath,backgroundpath)
    colors = getlegocolors(cents,cent_cols)
    recipe = character_map[character];
    #coordinates of the bricks required for building the character   
    coordinates_px = []
    # Loop though the recipe, and find the bricks needed
    for item in recipe:
        
        # Get index of the first colored brick in the list
        index = colors.index(item)
        
        # Set the brick to "used" such that it cannot be used any more
        colors[index] = '-';
        
        coordinates_px.append( cents[index] );
        
        
    return coordinates_px;

run( 'maggie', 'opencvnew0.png','opencvbgnew0.png');
